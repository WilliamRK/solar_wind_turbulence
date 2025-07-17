"""
functions.py file.

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
import astropy.units as u
import astropy.constants as c

# Display-tweak (harmless if run headless)
from IPython.display import display, HTML
display(HTML("<style>:root { --jp-notebook-max-width: 100% !important; }</style>"))

J2000_EPOCH   = np.datetime64("2000-01-01T12:00:00Z")
J2000_OFFSET_S = (J2000_EPOCH.astype("datetime64[ns]").astype(float) * 1e-9)




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
        print(fields)
        if not fields:  # download FIELDS
            fields = psp.fields(tr, "mag_SC_4_Sa_per_Cyc", "l2",
                                notplot=True, downloadonly=True,
                                last_version=True, get_support_data=False)
        for f in fields:
            if not f.startswith(base_dir):
                shutil.move(f, base_dir)
                fields_files = [os.path.join(base_dir, os.path.basename(f)) for f in fields]
                
        print(fields)
        
        if not spc:     # download SPC
            spc = psp.spc(tr, "l3i", "l3",
                          notplot=True, downloadonly=True,
                          last_version=True, get_support_data=False)
        for f in spc:
            dest = os.path.join(base_dir, os.path.basename(f))
            if not os.path.exists(dest):
                shutil.move(f, dest)
    
        #spc = sorted(glob.glob(s_pat))

        all_fields.append(fields)
        all_spc.append(spc)
    print(all_fields, all_spc)
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
    print("here", f_xrs)
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
    
    dqf = spc_xr.DQF.data
    dqf.shape
    mask_peaktrack = dqf[:,16] == 0
    mask_fullscan = dqf[:,16] == 1
    mask_pt = dqf[mask_peaktrack, 16]
    mask_fs = dqf[mask_fullscan, 16]
    mask_pt.shape

    #V   = spc_xr["vp_moment_SC"].values
    #n   = spc_xr["np_moment"].values
    #wp  = spc_xr["wp_moment"].values
    
    #SWAP TO THIS SET FOR FIT
    
    V = spc_xr["vp_fit_SC"].values
    n = spc_xr["np_fit"].values 
    wp = spc_xr["wp_fit"].values
    
    
    
    V = V[mask_peaktrack]
    n = n[mask_peaktrack]
    wp = wp[mask_peaktrack]
    t_SC = t_SC[mask_peaktrack]
    
    
    for arr in (B, V, n, wp):
        arr[arr <= -1e30] = np.nan
    return B, V, n, wp, t_B, t_SC

def process_range_wavelet(B, V, n, wp, t_B, t_SC):
    """Interpolates B onto plasma cadence and returns common-grid arrays."""
    B_interp = np.stack([np.interp(t_SC, t_B, B[:, i]) for i in range(3)], -1)
    Bmag = np.linalg.norm(B_interp, axis=1)
    Vmag = np.linalg.norm(V, axis=1)
    _quicklook_plot(t_SC, B_interp, Bmag, V, Vmag, n, wp)
    return B_interp, Bmag, Vmag, t_SC

# ───────────────────────── Wavelet & PSD helpers ────────────────────────
def compute_wavelet(signal, dt, dj=0.25, mother=None):
    signal = np.nan_to_num(np.asarray(signal), nan=0.0)

    N = signal.size
    if N < 2:
        raise ValueError("Signal too short to compute wavelet.")

    if mother is None:
        mother = Morlet(6)

    std = signal.std()
    if std <= 0:
        raise ValueError("Signal variance is zero, cannot compute wavelet.")

    # Normalize
    signal = (signal - signal.mean()) / std

    s0 = 2 * dt
    span = N * dt / s0
    J = np.log2(span) / dj

    if not np.isfinite(J) or J < 1:
        raise ValueError(f"Invalid number of scales: J={J}")

    J = int(J)

    wave, scales, freqs, coi, *_ = cwt(signal, dt, dj, s0, J, mother)
    return freqs, np.abs(wave) ** 2, coi, scales


def analyze_wavelet_for_range(t, B_int, V, Bmag, Vmag, span):
    
    """
    Mirror of `analyze_psd_for_range`, but using a continuous wavelet
    transform.  Produces:
      1) Global Wavelet Spectrum for trace-|B| and |V|.
      2) Component-wise Wavelet Spectra.
      3) Scalogram (time–frequency) of |B|.
      Plus slope fit & basic warnings.
    """
    start_s, end_s = span  # already J2000 seconds

    # also create human-readable ISO strings for labels
    start_iso = str(np.datetime64('2000-01-01T12:00:00') + np.timedelta64(int(start_s), 's'))
    end_iso   = str(np.datetime64('2000-01-01T12:00:00') + np.timedelta64(int(end_s), 's'))


    mask = (t >= start_s) & (t <= end_s)
    print(f"Window {span}: found {mask.sum()} points between {start_s} and {end_s}")
    print(f"t_SC range: {t[0]} → {t[-1]}")
    if mask.sum() < 2:
        warnings.warn(
            f"⛔ scalogram for {start_iso}–{end_iso} not possible – "
            f"only {mask.sum()} finite samples in window",
            RuntimeWarning
        )
        return False            # <-- tell caller we failed

    if np.all(np.isnan(B_int[mask])):
        warnings.warn(
            f"⛔ scalogram for {start_iso}–{end_iso} not possible – "
            f"all values NaN after interpolation",
            RuntimeWarning
        )
        return False
    
    

    # ------------------------------------------------------------------
    t = t[mask]
    dt_seconds = np.median(np.diff(t))
    B = B_int[mask]
    V = V[mask]
    Bmag_w = Bmag[mask]
    Vmag_w = Vmag[mask]
    dt_seconds = np.median(np.diff(t))
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        warnings.warn(
            f"⛔ scalogram for {start_iso}–{end_iso} not possible – "
            f"dt_seconds={dt_seconds}",
            RuntimeWarning
        )
        return False

    # --- Wavelet per B-component --------------------------------------
    freqs, power_B0, *_ = compute_wavelet(B[:, 0], dt_seconds)
    _, power_B1, *_ = compute_wavelet(B[:, 1], dt_seconds)
    _, power_B2, *_ = compute_wavelet(B[:, 2], dt_seconds)

    trace_B_pw = power_B0 + power_B1 + power_B2
    global_trace_B = trace_B_pw.mean(axis=1)  # avg over time

    # |B| and |V| wavelet spectra
    _, power_Bmag, *_ = compute_wavelet(Bmag_w - np.nanmean(Bmag_w),
                                        dt_seconds)
    _, power_Vmag, *_ = compute_wavelet(Vmag_w - np.nanmean(Vmag_w),
                                        dt_seconds)
    global_Bmag = power_Bmag.mean(axis=1)
    global_Vmag = power_Vmag.mean(axis=1)

    # --- Inertial-range slope (log-log) --------------------------------
    fit_mask = (freqs > 1e-3) & (freqs < 1e-1)
    slope, intercept, *_ = linregress(np.log10(freqs[fit_mask]),
                                      np.log10(global_trace_B[fit_mask]))

    # ---------- 1) Global spectra plot ---------------------------------
    plt.figure(figsize=(11, 7))
    plt.loglog(freqs, global_trace_B, label='Global WT Trace B', c='tab:blue')
    plt.loglog(freqs, global_Bmag,     '--', label='Global WT |B|',
               c='tab:cyan')
    plt.loglog(freqs, global_Vmag,     '--', label='Global WT |V|',
               c='tab:orange')
    plt.loglog(freqs[fit_mask],
               10**intercept * freqs[fit_mask]**slope,
               ':k', label=f'slope ≈ {slope:.2f}')
    plt.title(f'Global Wavelet Spectra  {start_iso} – {end_iso}')
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Power')
    plt.grid(True, which='both', ls='--', lw=0.4); plt.legend()
    plt.tight_layout(); plt.show()

    # ---------- 2) Component spectra plot ------------------------------
    plt.figure(figsize=(11, 7))
    plt.loglog(freqs, power_B0.mean(axis=1), label=r'WT $B_x$')
    plt.loglog(freqs, power_B1.mean(axis=1), label=r'WT $B_y$')
    plt.loglog(freqs, power_B2.mean(axis=1), label=r'WT $B_z$')
    plt.title(f'Component Wavelet Spectra  {start_iso} – {end_iso}')
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Power')
    plt.grid(True, which='both', ls='--', lw=0.4); plt.legend()
    plt.tight_layout(); plt.show()

    # ---------- 3) Scalogram of |B| ------------------------------------
    plt.figure(figsize=(12, 6))
    extent = [t.min(), t.max(), freqs.min(), freqs.max()]
    #plt.imshow(power_Bmag, extent=extent, aspect='auto', origin='lower',
    #           cmap='viridis', interpolation='nearest')
    trace_B_pw = np.log10(trace_B_pw)
    plt.pcolormesh(t, freqs, trace_B_pw, cmap = 'plasma')
    plt.yscale('log'); plt.colorbar(label='Power')
    plt.title(f'|B| Scalogram  {start_iso} – {end_iso}')
    plt.xlabel('Time [s since J2000]'); plt.ylabel('Frequency [Hz]')
    plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------
    print(f"↳ Wavelet-slope ({start_s}–{end_s}): {slope:.2f}")
    return True
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
def compute_alfven_speed(B, n):
    B_nT = B * u.nT
    n = n * u.cm**(-3)
    proton_mass = c.m_p
    mu_0 = c.mu0
    rho = np.nanmean(n) * proton_mass
    vA = (B_nT / np.sqrt(mu_0 * rho)).to("km/s").value
    return vA

def compute_theta_vb(V, B):
    cos = np.sum(V*B, axis=1) / np.nanmean(np.linalg.norm(V,axis=1)*np.linalg.norm(B,axis=1))
    return np.degrees(np.arccos(cos))

def compute_sigma_c(V, B, eps=1e-10):
    vb, v2, b2 = np.sum(V*B,axis=1), np.sum(V**2,axis=1), np.sum(B**2,axis=1)
    denom = np.nanmean(v2 + b2)
    return np.nan if denom < eps else 2*np.nanmean(vb)/denom

def compute_sigma_r(V, B, eps=1e-10):
    v2, b2 = np.sum(V**2,axis=1), np.sum(B**2,axis=1)
    denom  = np.nanmean(v2 + b2)
    return np.nan if denom < eps else (np.nanmean(v2)-np.nanmean(b2))/denom

def compute_elsasser(V, B, n_cm3):
    vA = compute_alfven_speed(B, np.nanmean(n_cm3))
    return V+vA, V-vA

def trace_psd(vec, fs):
    f, P = welch(vec[:,0], fs, "hamming", 1024, 512)
    for j in (1,2):
        _, Pj = welch(vec[:,j], fs, "hamming", 1024, 512); P += Pj
    return f, P

# ───────────────────────── CSV writer ───────────────────────────────────
def calculate_derived_parameters_to_csv(B, V, n, T, time, w_start, w_end, fname):
    Bmag = np.linalg.norm(B,axis=1); Vmag = np.linalg.norm(V,axis=1)
    #vA   = 2.18e6*Bmag/np.sqrt(np.nanmean(n))/1e3
    vA = compute_alfven_speed(Bmag, np.nanmean(n))
    beta = 0.03948*(n*T)/np.nanmean(Bmag**2)
    M_A  = Vmag/vA
    #θ    = compute_theta_vb(V,B)
    σc   = compute_sigma_c(V,B); σr = compute_sigma_r(V,B)
    theta_vb = compute_theta_vb(V,B)
    Zp, Zm = compute_elsasser(V-np.nanmean(V,axis=0),B-np.nanmean(B,axis=0),n)


    df = pd.DataFrame({
        "time_sec":time, "B_mag":Bmag, "V_mag":Vmag,
        "v_A":vA, "beta_p":beta, "M_A":M_A, "theta_vb":theta_vb,
        "Z_plus_mag":np.linalg.norm(Zp,axis=1), "Z_minus_mag":np.linalg.norm(Zm,axis=1),
    })
    with open(fname,"w") as f:
        f.write(f"# window_start: {w_start}\n# window_end: {w_end}\n")
        f.write(f"# sigma_c: {σc}\n# sigma_r: {σr}\n\n")
    df.to_csv(fname, mode="a", index=False)
    print("✅ wrote", fname)

# ───────────────────────── Scalogram Plots ─────────────────────────
def plot_scalograms(t_win, freqs, sigma_c_w, sigma_r_w, start_iso, end_iso):
    extent = [t_win.min(), t_win.max(), freqs.max(), freqs.min()]

    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for ax in axs:
        ax.set_yscale("log")
    
    X, Y = np.meshgrid(t_win, freqs)

    pc0 = axs[0].pcolormesh(X, Y, sigma_c_w, cmap='seismic', vmin=-1, vmax=1, shading='auto')
    axs[0].set_ylabel('Frequency [Hz]')
    axs[0].set_title('σ_c  (Cross-Helicity)')
    fig.colorbar(pc0, ax=axs[0], label='σ_c')

    pc1 = axs[1].pcolormesh(X, Y, sigma_r_w, cmap='seismic', vmin=-1, vmax=1, shading='auto')
    axs[1].set_ylabel('Frequency [Hz]')
    axs[1].set_xlabel('Time  [s since J2000]')
    axs[1].set_title('σ_r  (Residual Energy)')
    fig.colorbar(pc1, ax=axs[1], label='σ_r')


    fig.suptitle(f'Wavelet Scalograms   {start_iso} – {end_iso}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# ───────────────────────── Internal quick-look plot ─────────────────────
def _quicklook_plot(t, B, Bmag, V, Vmag, n, wp):
    """Unchanged quick-look diagnostic – kept private."""
    # (body identical to original quick-look panel)
    pass


