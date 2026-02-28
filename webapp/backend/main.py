import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import wfdb
from wfdb import processing
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_index():
    return FileResponse(Path(__file__).parent / "../frontend/index.html")

# ── ECGNetRR Definition ────────────────────────────────────────────────────────
class ECGNetRR(nn.Module):
    def __init__(self, rr_dim=10):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(0.5)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.rr_fc1 = nn.Linear(rr_dim, 64)
        self.rr_fc2 = nn.Linear(64, 32)
        self.rr_drop = nn.Dropout(0.2)
        self.fc = nn.Linear(256 + 32 + 1, 2)

    def forward(self, x, rr, rr_valid):
        x = F.relu(self.conv1(x)); x = self.pool(x); x = self.drop(x)
        x = F.relu(self.conv2(x)); x = self.pool(x); x = self.drop(x)
        x = F.relu(self.conv3(x)); x = self.pool(x); x = self.drop(x)
        x = F.relu(self.conv4(x)); x = self.pool(x); x = self.drop(x)
        x = self.gap(x).squeeze(-1)
        rr = F.relu(self.rr_fc1(rr)); rr = self.rr_drop(rr)
        rr = F.relu(self.rr_fc2(rr))
        z = torch.cat([x, rr, rr_valid], dim=1)
        return self.fc(z)

# ── Preprocessing constants ────────────────────────────────────────────────────
LEAD_IDX  = 0
TARGET_FS = None
WIN_SEC   = 10
STRIDE_SEC = 5
LOW_HZ    = 0.5
HIGH_HZ   = 40.0
F_ORDER   = 4

def bandpass_filter(x, fs):
    x = np.asarray(x, dtype=np.float32)
    nyq = fs / 2.0
    b, a = butter(F_ORDER, [LOW_HZ / nyq, HIGH_HZ / nyq], btype="bandpass")
    return filtfilt(b, a, x).astype(np.float32)

def detect_rpeaks_simple(x, fs):
    x = np.asarray(x, dtype=np.float32)
    energy = np.abs(np.diff(x, prepend=x[0])) ** 2
    peaks, _ = find_peaks(energy, distance=int(0.25 * fs), height=np.percentile(energy, 95))
    return peaks.astype(np.int64)

def rr_features_from_peaks(r_peaks_win, fs):
    if r_peaks_win is None or len(r_peaks_win) < 3:
        return np.zeros(10, dtype=np.float32), 0.0
    rr = np.diff(r_peaks_win).astype(np.float32) / float(fs)
    if len(rr) < 2:
        return np.zeros(10, dtype=np.float32), 0.0
    drr = np.diff(rr)
    abs_drr = np.abs(drr)
    mean_rr = rr.mean()
    sdnn = rr.std()
    rmssd = float(np.sqrt(np.mean(drr ** 2))) if len(drr) else 0.0
    pnn50 = float(np.mean(abs_drr > 0.05)) if len(abs_drr) else 0.0
    cv = float(sdnn / (mean_rr + 1e-8))
    med_rr = float(np.median(rr))
    iqr_rr = float(np.percentile(rr, 75) - np.percentile(rr, 25))
    mad_rr = float(np.median(np.abs(rr - med_rr)))
    if len(rr) >= 3:
        tp = sum(1 for i in range(1, len(rr)-1)
                 if (rr[i] > rr[i-1] and rr[i] > rr[i+1]) or
                    (rr[i] < rr[i-1] and rr[i] < rr[i+1]))
        tpr = float(tp / (len(rr) - 2))
    else:
        tpr = 0.0
    rr_range = float(rr.max() - rr.min())
    return np.array([mean_rr, sdnn, rmssd, pnn50, cv, med_rr, iqr_rr, mad_rr, tpr, rr_range], dtype=np.float32), 1.0

# ── Model loading ──────────────────────────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else
                      'cuda' if torch.cuda.is_available() else 'cpu')
models = []
rr_mean = None
rr_scale = None

TOP_5_MODELS = [
    "best_rrcnn_fold_04_04126.pth",
    "best_rrcnn_fold_05_04746.pth",
    "best_rrcnn_fold_07_04936.pth",
    "best_rrcnn_fold_17_07910.pth",
    "best_rrcnn_fold_22_08434.pth",
]

def load_model():
    global models, rr_mean, rr_scale
    
    # Try deployment 'models' folder first, then local 'results' folder
    models_dir = Path(__file__).parent / "models"
    results_dir = Path(__file__).parent / "../../../results"
    
    search_dirs = [models_dir, results_dir]
    
    models = []
    for filename in TOP_5_MODELS:
        p = None
        for d in search_dirs:
            if (d / filename).exists():
                p = d / filename
                break
                
        if not p:
            print(f"Skipping {filename}: not found in expected directories")
            continue
            
        try:
            ckpt = torch.load(p, map_location=device, weights_only=True)
            m = ECGNetRR(rr_dim=ckpt.get("rr_dim", 10))
            m.load_state_dict(ckpt["model_state"])
            m.to(device).eval()
            models.append(m)
            rr_mean  = ckpt.get("rr_mean",  np.zeros(10))
            rr_scale = ckpt.get("rr_scale", np.ones(10))
            print(f"Loaded {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    return bool(models)

@app.on_event("startup")
async def startup_event():
    load_model()

# ── Shared preprocessing ───────────────────────────────────────────────────────
def _preprocess(files_data: dict):
    tmp_dir = Path("./tmp_ecg")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    record_name = None
    for filename, data in files_data.items():
        (tmp_dir / filename).write_bytes(data)
        if filename.endswith(".hea"):
            record_name = filename.replace(".hea", "")
    if not record_name:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError("Missing .hea file")

    sig, fields = wfdb.rdsamp(str(tmp_dir / record_name))
    fs = float(fields["fs"])
    x = sig[:, LEAD_IDX].astype(np.float32)

    if TARGET_FS is not None and float(TARGET_FS) != fs:
        x, _ = processing.resample_sig(x, fs, TARGET_FS)
        x = x.astype(np.float32)
        fs = float(TARGET_FS)

    x = bandpass_filter(x, fs)
    win_len = int(WIN_SEC * fs)
    stride  = int(STRIDE_SEC * fs)
    if len(x) < win_len:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise ValueError("Signal too short for windowing")

    starts     = np.arange(0, len(x) - win_len + 1, stride, dtype=np.int64)
    rpeaks_all = detect_rpeaks_simple(x, fs)

    rr_feat_list, rr_valid_list = [], []
    for s in starts:
        s0, s1 = int(s), int(s + win_len)
        mask = (rpeaks_all >= s0) & (rpeaks_all < s1)
        f, v = rr_features_from_peaks(rpeaks_all[mask] - s0, fs)
        rr_feat_list.append(f)
        rr_valid_list.append(v)

    rr_feat  = np.stack(rr_feat_list).astype(np.float32)
    rr_valid = np.asarray(rr_valid_list, dtype=np.float32)
    X  = np.stack([x[s:s + win_len] for s in starts]).astype(np.float32)
    Xn = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    rr_feat_sc = (rr_feat - rr_mean) / (rr_scale + 1e-8)
    rr_feat_sc[rr_valid <= 0.5] = 0.0

    shutil.rmtree(tmp_dir, ignore_errors=True)

    X_t  = torch.tensor(Xn,         dtype=torch.float32).unsqueeze(1).to(device)
    rr_t = torch.tensor(rr_feat_sc, dtype=torch.float32).to(device)
    rv_t = torch.tensor(rr_valid,   dtype=torch.float32).unsqueeze(1).to(device)
    return X_t, rr_t, rv_t, starts, fs

def _ensemble_probs(X_b, rr_b, rv_b):
    """Run one batch through all 5 models and return averaged probabilities."""
    with torch.no_grad():
        return np.mean([
            F.softmax(m(X_b, rr_b, rv_b), dim=1)[:, 1].cpu().numpy()
            for m in models
        ], axis=0)

# ── Streaming endpoint ─────────────────────────────────────────────────────────
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",   # disables nginx buffering if behind a proxy
    "Connection": "keep-alive",
}

@app.post("/api/predict/stream")
async def predict_stream(files: list[UploadFile] = File(...)):
    if not models and not load_model():
        async def err():
            yield f"data: {json.dumps({'type':'error','message':'Models not loaded'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream", headers=SSE_HEADERS)

    files_data = {f.filename: await f.read() for f in files}

    async def event_stream():
        try:
            X_t, rr_t, rv_t, starts, fs = _preprocess(files_data)
            total = len(starts)
            yield f"data: {json.dumps({'type':'init','total':total})}\n\n"

            BATCH = 64
            all_preds, all_probs = [], []
            # per-model running AF window count
            model_af_counts = [0] * len(models)

            for bs in range(0, total, BATCH):
                be = min(bs + BATCH, total)
                X_b  = X_t[bs:be]
                rr_b = rr_t[bs:be]
                rv_b = rv_t[bs:be]

                # Get individual model probabilities
                with torch.no_grad():
                    per_model_probs = []
                    for m in models:
                        p = F.softmax(m(X_b, rr_b, rv_b), dim=1)[:, 1].cpu().numpy()
                        per_model_probs.append([round(float(x), 4) for x in p])

                avg   = np.mean([np.array(p) for p in per_model_probs], axis=0)
                preds = (avg >= 0.5).astype(int).tolist()
                probs = [round(float(p), 4) for p in avg]
                all_preds.extend(preds)
                all_probs.extend(probs)

                # Accumulate per-model AF counts
                for mi, mp in enumerate(per_model_probs):
                    model_af_counts[mi] += sum(1 for p in mp if p >= 0.5)

                # Compute batch-level per-model summary (avg prob this batch, af vote rate so far)
                model_summary = [
                    {
                        "batch_avg_prob": round(float(np.mean(per_model_probs[mi])), 4),
                        "running_af_pct": round(model_af_counts[mi] / be * 100, 1),
                    }
                    for mi in range(len(models))
                ]

                yield f"data: {json.dumps({'type':'progress','processed':be,'total':total,'probs':probs,'preds':preds,'batch_start':bs,'model_summary':model_summary})}\n\n"
                await asyncio.sleep(0)  # yield to event loop so the chunk flushes immediately

            af  = int(sum(all_preds))
            pct = round(af / total * 100, 2) if total else 0.0
            yield f"data: {json.dumps({'type':'complete','total_windows':total,'af_windows':af,'normal_windows':total-af,'percent_af':pct,'diagnosis':'Atrial Fibrillation Detected' if pct>=20 else 'Normal Sinus Rhythm'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)

# ── Non-streaming endpoint (kept for compatibility) ────────────────────────────
@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...)):
    if not models and not load_model():
        return {"error": "Models not loaded on server."}
    files_data = {f.filename: await f.read() for f in files}
    try:
        X_t, rr_t, rv_t, starts, fs = _preprocess(files_data)
        avg_probs = _ensemble_probs(X_t, rr_t, rv_t)
        preds = (avg_probs >= 0.5).astype(int)
        total  = len(preds)
        af     = int(preds.sum())
        pct    = round(float(af / total) * 100, 2)
        return {
            "total_windows": total, "af_windows": af,
            "normal_windows": total - af, "percent_af": pct,
            "diagnosis": "Atrial Fibrillation Detected" if pct >= 20 else "Normal Sinus Rhythm"
        }
    except Exception as e:
        return {"error": str(e)}

# ── Sample records ─────────────────────────────────────────────────────────────
SAMPLES_DIR = Path(__file__).parent / "samples"
SAMPLE_RECORDS = {
    "04015": {"label": "Patient 04015", "desc": "Predominantly normal rhythm (~0.6% AFib)"},
    "04043": {"label": "Patient 04043", "desc": "Moderate AFib episodes (~21% AFib)"},
    "04936": {"label": "Patient 04936", "desc": "Heavy AFib burden (~81% AFib)"},
}

@app.get("/api/samples")
async def list_samples():
    return [{"id": k, **v} for k, v in SAMPLE_RECORDS.items()]

def _preprocess_from_disk(record_path: str):
    """Same as _preprocess but reads directly from a file path instead of uploaded bytes."""
    sig, fields = wfdb.rdsamp(record_path)
    fs = float(fields["fs"])
    x = sig[:, LEAD_IDX].astype(np.float32)

    if TARGET_FS is not None and float(TARGET_FS) != fs:
        x, _ = processing.resample_sig(x, fs, TARGET_FS)
        x = x.astype(np.float32)
        fs = float(TARGET_FS)

    x = bandpass_filter(x, fs)
    win_len = int(WIN_SEC * fs)
    stride  = int(STRIDE_SEC * fs)
    if len(x) < win_len:
        raise ValueError("Signal too short for windowing")

    starts     = np.arange(0, len(x) - win_len + 1, stride, dtype=np.int64)
    rpeaks_all = detect_rpeaks_simple(x, fs)

    rr_feat_list, rr_valid_list = [], []
    for s in starts:
        s0, s1 = int(s), int(s + win_len)
        mask = (rpeaks_all >= s0) & (rpeaks_all < s1)
        f, v = rr_features_from_peaks(rpeaks_all[mask] - s0, fs)
        rr_feat_list.append(f)
        rr_valid_list.append(v)

    rr_feat  = np.stack(rr_feat_list).astype(np.float32)
    rr_valid = np.asarray(rr_valid_list, dtype=np.float32)
    X  = np.stack([x[s:s + win_len] for s in starts]).astype(np.float32)
    Xn = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    rr_feat_sc = (rr_feat - rr_mean) / (rr_scale + 1e-8)
    rr_feat_sc[rr_valid <= 0.5] = 0.0

    X_t  = torch.tensor(Xn,         dtype=torch.float32).unsqueeze(1).to(device)
    rr_t = torch.tensor(rr_feat_sc, dtype=torch.float32).to(device)
    rv_t = torch.tensor(rr_valid,   dtype=torch.float32).unsqueeze(1).to(device)
    return X_t, rr_t, rv_t, starts, fs

@app.get("/api/predict/sample/stream/{record_id}")
async def predict_sample_stream(record_id: str):
    if record_id not in SAMPLE_RECORDS:
        async def err():
            yield f"data: {json.dumps({'type':'error','message':'Unknown sample record'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream", headers=SSE_HEADERS)

    if not models and not load_model():
        async def err2():
            yield f"data: {json.dumps({'type':'error','message':'Models not loaded'})}\n\n"
        return StreamingResponse(err2(), media_type="text/event-stream", headers=SSE_HEADERS)

    record_path = str(SAMPLES_DIR / record_id)

    async def event_stream():
        try:
            X_t, rr_t, rv_t, starts, fs = _preprocess_from_disk(record_path)
            total = len(starts)
            yield f"data: {json.dumps({'type':'init','total':total})}\n\n"

            BATCH = 64
            all_preds, all_probs = [], []
            model_af_counts = [0] * len(models)

            for bs in range(0, total, BATCH):
                be = min(bs + BATCH, total)
                X_b, rr_b, rv_b = X_t[bs:be], rr_t[bs:be], rv_t[bs:be]

                with torch.no_grad():
                    per_model_probs = []
                    for m in models:
                        p = F.softmax(m(X_b, rr_b, rv_b), dim=1)[:, 1].cpu().numpy()
                        per_model_probs.append([round(float(x), 4) for x in p])

                avg   = np.mean([np.array(p) for p in per_model_probs], axis=0)
                preds = (avg >= 0.5).astype(int).tolist()
                probs = [round(float(p), 4) for p in avg]
                all_preds.extend(preds)
                all_probs.extend(probs)

                for mi, mp in enumerate(per_model_probs):
                    model_af_counts[mi] += sum(1 for p in mp if p >= 0.5)

                model_summary = [
                    {
                        "batch_avg_prob": round(float(np.mean(per_model_probs[mi])), 4),
                        "running_af_pct": round(model_af_counts[mi] / be * 100, 1),
                    }
                    for mi in range(len(models))
                ]

                yield f"data: {json.dumps({'type':'progress','processed':be,'total':total,'probs':probs,'preds':preds,'batch_start':bs,'model_summary':model_summary})}\n\n"
                await asyncio.sleep(0)

            af  = int(sum(all_preds))
            pct = round(af / total * 100, 2) if total else 0.0
            yield f"data: {json.dumps({'type':'complete','total_windows':total,'af_windows':af,'normal_windows':total-af,'percent_af':pct,'diagnosis':'Atrial Fibrillation Detected' if pct>=20 else 'Normal Sinus Rhythm'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)

