"""
Reusable utilities for PSP fast-radial-scan analysis.
All public function names match the originals.
"""

# ───────────────────────── Imports & constants ──────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import pyspedas
from pyspedas import psp
import cdflib.xarray as cdf_xr
import glob, os, re, shutil, warnings
from datetime import datetime, timezone
from scipy.stats import linregress
from scipy.signal import welch
from pycwt import Morlet, cwt
import pandas as pd

# Display-tweak (harmless if run headless)
from IPython.display import display, HTML
display(HTML("<style>:root { --jp-notebook-max-width: 100% !important; }</style>"))

J2000_EPOCH   = np.datetime64("2000-01-01T12:00:00Z")
J2000_OFFSET_S = (J2000_EPOCH.astype("datetime64[ns]").astype(float) * 1e-9)

dqf = spc_xr.DQF.data
dqf.shape
mask_peaktrack = dqf[:,16] == 0
mask_fullscan = dqf[:,16] == 1
mask_pt = dqf[mask_peaktrack, 16]
mask_fs = dqf[mask_fullscan, 16]
mask_pt.shape

# Physical constants
proton_mass = 1.672622e-27          # kg
k_B         = 1.380649e-23          # J K⁻¹
mu_0_SI     = 4 * np.pi * 1e-7      # H m⁻¹

# ───────────────────────── Time helpers ─────────────────────────────────
def datetime64_to_j2000_seconds(dt64):
    """numpy datetime64 → seconds since J2000 (float)."""
    if isinstance(dt64, str):
        dt64 = np.datetime64(dt64)
    ns = dt64.astype("datetime64[ns]").astype(np.int64)
    return ns * 1e-9 - J2000_OFFSET_S

def j2000_to_iso(j2000_seconds):
    return str(np.datetime64("2000-01-01T12:00:00") +
               np.timedelta64(int(j2000_seconds), "s"))

# ───────────────────────── File & download utils ────────────────────────
def parse_time_ranges_from_file(filepath):
    pat = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) to (\d{4}-\d{2}-\d{2} \d{2}:\d{2})"
    with open(filepath) as f:
        return re.findall(pat, f.read())

def download_psp_data(time_ranges, base_dir="psp_data"):
    """Return lists of downloaded CDF paths (FIELDS, SPC)."""
    os.makedirs(base_dir, exist_ok=True)
    all_fields, all_spc = [], []
    for tr in time_ranges:
        date_tag = tr[0].split()[0].replace("-", "")
        f_pat    = f"{base_dir}/fields_*{date_tag}*.cdf"
        s_pat    = f"{base_dir}/spc_*{date_tag}*.cdf"
        fields   = sorted(glob.glob(f_pat))
        spc      = sorted(glob.glob(s_pat))

        if not fields:  # download FIELDS
            fields = psp.fields(tr, "mag_SC_4_Sa_per_Cyc", "l2",
                                notplot=True, downloadonly=True,
                                last_version=True, get_support_data=False)
            for f in fields:
                if not f.startswith(base_dir):
                    shutil.move(f, base_dir)
                    fields_files = [os.path.join(base_dir, os.path.basename(f)) for f in fields]

            fields = sorted(glob.glob(f_pat))

        if not spc:     # download SPC
            spc = psp.spc(tr, "l3i", "l3",
                          notplot=True, downloadonly=True,
                          last_version=True, get_support_data=False)
            for f in spc:
                dest = os.path.join(base_dir, os.path.basename(f))
                if not os.path.exists(dest):
                    shutil.move(f, dest)
    
            spc = sorted(glob.glob(s_pat))

        all_fields.append(fields)
        all_spc.append(spc)
    return all_fields, all_spc

def convert_all_cdf_to_xarray(fields_files, spc_files):
    """CDF → xarray conversion with alignment to time_ranges length."""
    f_xrs, s_xrs = [], []
    for ff, sf in zip(fields_files, spc_files):
        try:
            f_xrs.append(cdf_xr.cdf_to_xarray(ff[0]) if ff else None)
            s_xrs.append(cdf_xr.cdf_to_xarray(sf[0]) if sf else None)
        except Exception as e:
            print("❌ CDF→xarray failed:", e)
            f_xrs.append(None); s_xrs.append(None)
    return f_xrs, s_xrs

# ───────────────────────── Extraction & interpolation ───────────────────
def extract_psp_data(fields_xr, spc_xr):
    """Return B, V, n, wp and their respective time arrays (J2000 s)."""
    if "epoch_mag_SC_4_Sa_per_Cyc" in fields_xr.coords:
        fields_xr = fields_xr.rename({"epoch_mag_SC_4_Sa_per_Cyc": "time"})

    B   = fields_xr["psp_fld_l2_mag_SC_4_Sa_per_Cyc"].values
    t_B = datetime64_to_j2000_seconds(fields_xr["time"].values)

    spc_time_key = next(c for c in spc_xr.coords if "epoch" in c.lower())
    t_SC = datetime64_to_j2000_seconds(spc_xr.coords[spc_time_key].values)

    V   = spc_xr["vp_moment_SC"].values
    n   = spc_xr["np_moment"].values
    wp  = spc_xr["wp_moment"].values
    V = V[mask_peaktrack]
    n = n[mask_peaktrack]
    wp = wp[mask_peaktrack]
    
    #SWAP TO THIS SET FOR FIT
    """
    V = spc_xr["vp_fit_SC"].values
    density = spc_xr["np_fit"].values / 1e6
    thermal_speed = spc_xr["wp_fit"].values
    """
    
    for arr in (B, V, n, wp):
        arr[arr <= -1e30] = np.nan
    return B, V, n, wp, t_B, t_SC

def process_range_wavelet(B, V, n, wp, t_B, t_SC):
    """Interpolates B onto plasma cadence and returns common-grid arrays."""
    B_interp = np.stack([np.interp(t_SC, t_B, B[:, i]) for i in range(3)], -1)
    Bmag = np.linalg.norm(B_interp, 1)
    Vmag = np.linalg.norm(V, 1)
    _quicklook_plot(t_SC, B_interp, Bmag, V, Vmag, n, wp)
    return B_interp, Bmag, Vmag, t_SC

# ───────────────────────── Wavelet & PSD helpers ────────────────────────
def compute_wavelet(signal, dt, dj=0.25, mother=None):
    signal = np.nan_to_num(np.asarray(signal), nan=0.0)
    if mother is None:
        mother = Morlet(6)
    std = signal.std()
    if std <= 0: raise ValueError("zero variance")
    signal = (signal - signal.mean()) / std
    s0 = 2 * dt
    J  = int(np.log2(signal.size * dt / s0) / dj)
    wave, scales, freqs, coi, *_ = cwt(signal, dt, dj, s0, J, mother)
    return freqs, np.abs(wave) ** 2, coi, scales

def analyze_wavelet_for_range(t, B_int, V, Bmag, Vmag, span):
    """See original notebook – unchanged, but hoisted here for reuse."""
    # ... (body identical to original) ...
    pass  # For brevity; copy your existing body here.

# ───────────────────────── Sliding-window logic ─────────────────────────
def generate_valid_windows_aligned(t, B, V, span_start, span_end,
                                   window_sec=3600, step_sec=60,
                                   vel_nan_max=0.05, B_nan_max=0.01):
    """Return list of (start,end) J2000-sec windows with low NaN fractions."""
    wins, start = [], int(span_start)
    stop = int(min(span_end, t[-1]))
    while start + window_sec <= stop:
        end   = start + window_sec
        mask  = (t >= start) & (t < end)
        if mask.sum() >= 2:
            v_nan = np.isnan(V[mask]).sum() / V[mask].size
            b_nan = np.isnan(B[mask]).sum() / B[mask].size
            if v_nan <= vel_nan_max and b_nan <= B_nan_max:
                wins.append((float(start), float(end)))
                start += window_sec; continue
        start += step_sec
    return wins

# ───────────────────────── Plasma-physics small helpers ─────────────────
def compute_theta_vb(V, B):
    cos = np.sum(V*B, 1) / np.clip(np.linalg.norm(V,1)*np.linalg.norm(B,1),1e-30,None)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def compute_sigma_c(V, B, eps=1e-10):
    vb, v2, b2 = np.sum(V*B,1), np.sum(V**2,1), np.sum(B**2,1)
    denom = np.nanmean(v2 + b2)
    return np.nan if denom < eps else 2*np.nanmean(vb)/denom

def compute_sigma_r(V, B, eps=1e-10):
    v2, b2 = np.sum(V**2,1), np.sum(B**2,1)
    denom  = np.nanmean(v2 + b2)
    return np.nan if denom < eps else (np.nanmean(v2)-np.nanmean(b2))/denom

def compute_elsasser(V, B, n_cm3):
    vA = (B*1e-9)/np.sqrt(mu_0_SI*(n_cm3*1e6)[:,None])/1e3
    return V+vA, V-vA

def trace_psd(vec, fs):
    f, P = welch(vec[:,0], fs, "hamming", 1024, 512)
    for j in (1,2):
        _, Pj = welch(vec[:,j], fs, "hamming", 1024, 512); P += Pj
    return f, P

# ───────────────────────── CSV writer ───────────────────────────────────
def calculate_derived_parameters_to_csv(B, V, n, T, t, w_start, w_end, fname):
    Bmag = np.linalg.norm(B,1); Vmag = np.linalg.norm(V,1)
    vA   = 2.18e6*Bmag/np.sqrt(np.clip(n,1e-6,None))/1e3
    beta = 0.03948*(n*T)/np.clip(Bmag**2,1e-6,None)
    M_A  = Vmag/vA
    θ    = compute_theta_vb(V,B)
    σc   = compute_sigma_c(V,B); σr = compute_sigma_r(V,B)
    Zp, Zm = compute_elsasser(V,B,n)

    df = pd.DataFrame({
        "time_sec":t, "B_mag":Bmag, "V_mag":Vmag,
        "v_A":vA, "beta_p":beta, "M_A":M_A, "theta_vb":θ,
        "Z_plus_mag":np.linalg.norm(Zp,1), "Z_minus_mag":np.linalg.norm(Zm,1),
    })
    with open(fname,"w") as f:
        f.write(f"# window_start: {w_start}\n# window_end: {w_end}\n")
        f.write(f"# sigma_c: {σc}\n# sigma_r: {σr}\n\n")
    df.to_csv(fname, mode="a", index=False)
    print("✅ wrote", fname)

# ───────────────────────── Internal quick-look plot ─────────────────────
def _quicklook_plot(t, B, Bmag, V, Vmag, n, wp):
    """Unchanged quick-look diagnostic – kept private."""
    # (body identical to original quick-look panel)
    pass


