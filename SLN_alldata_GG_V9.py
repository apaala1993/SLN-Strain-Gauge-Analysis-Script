
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks, peak_widths
import matplotlib
matplotlib.rcParams["agg.path.chunksize"] = 10000
matplotlib.rcParams["path.simplify"] = True
matplotlib.rcParams["path.simplify_threshold"] = 1.0
matplotlib.rcParams["axes.ymargin"] = 0.15
matplotlib.rcParams["axes.grid"] = False
import matplotlib.pyplot as plt

# =============================================================================
# 0. Configuration
# =============================================================================

th = 0.002
tbefore_sec = 5.0
tafter_sec = 5.0
stimint_sec = 10.0

stim_names = ["BS1", "A10", "A20", "A30", "A40", "BS2",
              "D40", "D30", "D20", "D10", "BS3"]

TARGET_CONDITIONS = ["A30", "A40", "D30", "D40"]
BASELINE_CONDITIONS = ["BS1", "BS2", "BS3"]

calibration_factor = 1.66 / 5

# SG exclusion window after stimulation onset (seconds).
# Set anywhere from 0 to 20 seconds.
TIME_EXCLUDE = 2.0
if not 0 <= TIME_EXCLUDE <= 20:
    raise ValueError("TIME_EXCLUDE must be between 0 and 20 seconds.")

# Number of highest-amplitude counted SG peaks to include in amplitude/force summaries.
# Set to 0 to include ALL counted peaks; otherwise use the top N peaks.
NUMBER_PEAK_INCLUDE = 10
if not isinstance(NUMBER_PEAK_INCLUDE, (int, np.integer)) or NUMBER_PEAK_INCLUDE < 0:
    raise ValueError("NUMBER_PEAK_INCLUDE must be a non-negative integer (0 = all peaks).")

# --- SG baseline (pre-stim window) ---
# A pre-stim bump is treated as a peak (and excluded from the baseline) if its
# prominence is at least this fraction of the swallow-detection threshold.
BASELINE_PEAK_PROMINENCE_FRACTION = 0.5
# How much of each such peak is cut out: its width at this fraction of its prominence.
BASELINE_PEAK_WIDTH_REL_HEIGHT = 0.95

# --- Swallow latency ---
MESSY_SIGNAL_WINDOW_SEC = 2.0

# =============================================================================
# RESPIRATORY (RR) USER INPUTS
# Every RR peak is judged by its PROMINENCE (how far it rises above the dips on
# either side of it) compared with a NORMAL BREATH measured in the pre-stim
# window. The first three inputs are prominence cut-offs; the rest are timings.
# =============================================================================

# --- PROMINENCE inputs (fractions; decide what a peak is) ---

BREATH_REFERENCE_PROMINENCE_FRACTION = 0.3

BREATH_MIN_PROMINENCE_FRACTION = 0.5

BLIP_MIN_PROMINENCE_FRACTION = 0.25

BREATH_MIN_SPACING_FRACTION = 0.4

EXTENDED_APNEA_MIN_SEC = 1.5            # seconds

EXTENDED_APNEA_IBI_MULTIPLIER = 3.0     # x normal pre-stim breath interval

VERIFIED_APNEA_MULTIPLIER = 1.5

BREATH_CYCLE_BASELINE_FRACTION = 0.1

for _name, _val in (("BREATH_CYCLE_BASELINE_FRACTION", BREATH_CYCLE_BASELINE_FRACTION),
                    ("BREATH_REFERENCE_PROMINENCE_FRACTION", BREATH_REFERENCE_PROMINENCE_FRACTION),
                    ("BREATH_MIN_PROMINENCE_FRACTION", BREATH_MIN_PROMINENCE_FRACTION),
                    ("BLIP_MIN_PROMINENCE_FRACTION", BLIP_MIN_PROMINENCE_FRACTION)):
    if not (0 < _val <= 1):
        raise ValueError(f"{_name} must be between 0 and 1 (got {_val}).")
if BLIP_MIN_PROMINENCE_FRACTION >= BREATH_MIN_PROMINENCE_FRACTION:
    raise ValueError("BLIP_MIN_PROMINENCE_FRACTION must be smaller than BREATH_MIN_PROMINENCE_FRACTION.")

# Swallow-Breathing Coordination (SBC): the swallow-breath lag (swallow SG peak minus
# the RR peak of its respiratory cycle, in ms). Excel sheet names are limited to 31
# characters, so the per-animal sheet is "Swallow-Breathing Coord (SBC)".
SBC_SHEET = "Swallow-Breathing Coord (SBC)"
SBC_VALUES_SHEET = "SBC Values"
SBC_SUMMARY_COLS = ["N_Swallows_Lag_Measured", "Swallow_RR_Lag_mean_ms", "Swallow_RR_Lag_median_ms",
                    "N_Lag_Positive", "Pct_Lag_Positive", "N_Lag_Negative", "Pct_Lag_Negative", "N_Lag_Zero",
                    "Lag_Positive_values_ms", "Lag_Positive_mean_ms", "Lag_Positive_median_ms",
                    "Lag_Negative_values_ms", "Lag_Negative_mean_ms", "Lag_Negative_median_ms",
                    "N_Swallows_Outside_Resp_Cycle"]
_SBC_TOKENS = [("Swallow-breath lag", "Swallow-Breathing Coordination (SBC)"),
               ("swallow-breath lag", "Swallow-Breathing Coordination (SBC)"),
               ("N_Swallows_Lag_Measured", "SBC_N_Measured"), ("Swallow_RR_Lag", "SBC"),
               ("Pct_Lag_Positive", "SBC_Pct_Positive"), ("Pct_Lag_Negative", "SBC_Pct_Negative"),
               ("N_Lag_Positive", "SBC_N_Positive"), ("N_Lag_Negative", "SBC_N_Negative"), ("N_Lag_Zero", "SBC_N_Zero"),
               ("Lag_Positive_", "SBC_Positive_"), ("Lag_Negative_", "SBC_Negative_"), ("Lag_sign", "SBC_sign")]


def _sbc_rename(text):
    """Apply the SBC naming to a column label or a README text."""
    if not isinstance(text, str):
        return text
    for old, new in _SBC_TOKENS:
        text = text.replace(old, new)
    return text


APNEA_LABEL_SWALLOW_RR = "Swallow-RR Apnea"
APNEA_LABEL_STIM_RR = "Stim-RR Apnea"

# =============================================================================
# 1. Helper functions
# =============================================================================

def _comtext_to_strings(comtext_raw):
    arr = np.array(comtext_raw)
    if arr.dtype.kind in ("U", "S"):
        rows = []
        for row in arr:
            if isinstance(row, bytes):
                rows.append(row.decode("utf-8", errors="ignore").rstrip("\x00").rstrip())
            else:
                rows.append(str(row).rstrip())
        return rows
    rows = []
    for row in arr:
        s = "".join(chr(int(c)) for c in np.atleast_1d(row) if c > 0)
        rows.append(s.rstrip())
    return rows


def load_channel(matfile):
    m = loadmat(matfile, squeeze_me=True, struct_as_record=False)
    data = np.atleast_1d(np.asarray(m["data"])).ravel()
    samplerate = np.atleast_1d(np.array(m["samplerate"], dtype=float)).ravel()
    out = {"data": data, "samplerate": samplerate}
    for key in ("com", "comtext", "datastart", "dataend"):
        if key in m:
            out[key] = m[key]
    return out


def find_stim_events(stim, fs, th, stimint_sec):
    stimint = 0.8 * stimint_sec * fs
    inds = np.flatnonzero(stim > th)
    if inds.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    dinds = np.diff(inds) > stimint
    dinds = np.roll(dinds, 1)
    dinds[0] = False
    startinds = np.concatenate(([inds[0]], inds[1:][dinds]))
    stopinds = np.concatenate((inds[:-1][dinds], [inds[-1]]))
    return startinds, stopinds


def _otsu_threshold(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2 or np.ptp(values) == 0:
        return float(values.max()) if values.size else 0.0
    hist, bin_edges = np.histogram(values, bins=min(256, max(10, values.size)))
    hist = hist.astype(float)
    prob = hist / hist.sum()
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(prob * bin_mids)
    global_mean = cum_mean[-1]
    denom = cum_prob * (1 - cum_prob)
    with np.errstate(invalid="ignore", divide="ignore"):
        between_var = (global_mean * cum_prob - cum_mean) ** 2 / denom
    between_var[~np.isfinite(between_var)] = -1
    return float(bin_mids[np.argmax(between_var)])


PROMINENCE_SAFETY_MARGIN = 0.9


def _auto_peak_params(signal):
    all_locs, all_props = find_peaks(signal, prominence=1e-9)
    if all_locs.size < 2:
        return 0.0, 1
    prominences = all_props["prominences"]
    prom_threshold = _otsu_threshold(prominences) * PROMINENCE_SAFETY_MARGIN
    kept = all_locs[prominences >= prom_threshold]
    if kept.size >= 2:
        spacing = np.diff(kept)
        min_distance = max(1, int(round(0.35 * np.median(spacing))))
    else:
        min_distance = 1
    return prom_threshold, min_distance


def _decimate_for_plot(x, y, max_points=15000):
    n = len(x)
    if n <= max_points:
        return x, y
    step = int(np.ceil(n / max_points))
    return x[::step], y[::step]


def process_animal(data_folder, file_prefix, animal_id):
    DATA_FOLDER = data_folder
    FILE_PREFIX = file_prefix

    SAVE_FIGURES = True
    SHOW_FIGURES = False
    FIGURE_OUTPUT_DIR = os.path.join(DATA_FOLDER, "figures_output")
    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

    _fig_counter = {"n": 0}

    def finish_figure(fig, name):
        _fig_counter["n"] += 1
        if SAVE_FIGURES:
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            path = os.path.join(FIGURE_OUTPUT_DIR, f"{_fig_counter['n']:03d}_{safe_name}.png")
            fig.savefig(path, dpi=120)
        if not SHOW_FIGURES:
            plt.close(fig)

    def _f(suffix):
        return os.path.join(DATA_FOLDER, FILE_PREFIX + suffix)

    # =============================================================================
    # 2. Load channel files
    # =============================================================================

    required_channels = [
        ("Ch2 STIM.mat", "stim"), ("Ch1 RR.mat", "RR"),
        ("Ch3 SG.mat", "SG"), ("Ch9 EMG RMS.mat", "EMG RMS"),
    ]
    for suffix, label in required_channels:
        expected_path = _f(suffix)
        if not os.path.isfile(expected_path):
            print(f"  WARNING: Rat {animal_id} is missing the {label} channel file "
                  f"(expected: {os.path.basename(expected_path)}) -- SKIPPING this animal.")
            return None

    try:
        stim_ch = load_channel(_f("Ch2 STIM.mat"))
        stim = stim_ch["data"]
        fs = stim_ch["samplerate"][0]

        rr_ch = load_channel(_f("Ch1 RR.mat"))
        RR = rr_ch["data"]
        fsRR = rr_ch["samplerate"][0]

        sg_ch = load_channel(_f("Ch3 SG.mat"))
        SG = sg_ch["data"]

        rms_ch = load_channel(_f("Ch9 EMG RMS.mat"))
        RMS = rms_ch["data"]
    except Exception as e:
        print(f"  WARNING: Rat {animal_id} failed to load channel data ({e}) -- SKIPPING this animal.")
        return None

    comtext_raw = rms_ch["comtext"]
    com = np.array(rms_ch["com"], dtype=float)
    datastart = np.atleast_1d(np.array(rms_ch["datastart"], dtype=float)).ravel()

    fcoef = fs / fsRR

    # =============================================================================
    # 3. Parse comment text -> which comment indices belong to which condition
    # =============================================================================

    comtext = _comtext_to_strings(comtext_raw)
    ncomtext = len(comtext)

    stim_codes = {name: [] for name in stim_names}

    for i in range(ncomtext):
        text = comtext[i]
        comment_index = i + 1

        if "BASELINE 1" in text:
            if "START" in text:
                stim_codes["BS1"] = [comment_index]
            if "END" in text:
                stim_codes["BS1"] = stim_codes["BS1"] + [comment_index]
        if "BASELINE 2" in text:
            if "START" in text:
                stim_codes["BS2"] = [comment_index]
            if "END" in text:
                stim_codes["BS2"] = stim_codes["BS2"] + [comment_index]
        if "BASELINE 3" in text:
            if "START" in text:
                stim_codes["BS3"] = [comment_index]
            if "END" in text:
                stim_codes["BS3"] = stim_codes["BS3"] + [comment_index]
        for name in ("A10", "A20", "A30", "A40", "D40", "D30", "D20", "D10"):
            if text.strip() == name:
                stim_codes[name].append(comment_index)

    missing_conditions = []
    for name in BASELINE_CONDITIONS:
        if len(stim_codes[name]) < 2:
            missing_conditions.append(name)
    for name in TARGET_CONDITIONS:
        if len(stim_codes[name]) == 0:
            missing_conditions.append(name)
    if missing_conditions:
        print(f"  WARNING: Rat {animal_id} is missing expected recording(s) for: "
              f"{', '.join(missing_conditions)}. Continuing with whatever data IS present "
              f"-- check this animal's output carefully.")

    # =============================================================================
    # 4. Build `coms`: (code, absolute sample index) for every comment marker
    # =============================================================================

    ncoms = com.shape[0]
    coms = np.zeros((ncoms, 2))
    coms[:, 0] = com[:, 4]
    for i in range(ncoms):
        blocknum = int(com[i, 1])
        coms[i, 1] = datastart[blocknum - 1] + com[i, 2]

    # =============================================================================
    # 5. Detect stimulus on/off periods on the raw `stim` trace
    # =============================================================================

    stimint_samples = 0.8 * stimint_sec * fs
    tbefore_samples = tbefore_sec * fs
    tafter_samples = tafter_sec * fs

    startinds, stopinds = find_stim_events(stim, fs, th, stimint_sec)

    # =============================================================================
    # 6. Segment each channel (SG, RR, EMG RMS) into per-condition entries. EMG
    # =============================================================================

    stimtype = {name: {"codes": stim_codes[name], "stim": np.array([]),
                        "sg": np.array([]), "rr": np.array([]), "emgrms": np.array([])}
                for name in stim_names}

    for name in stim_names:
        entry = stimtype[name]
        if name in ("BS1", "BS2", "BS3"):
            if len(entry["codes"]) < 2:
                continue
            startcode_marker, stopcode_marker = entry["codes"][0], entry["codes"][1]
            startcode_row = np.flatnonzero(coms[:, 0] == startcode_marker)
            stopcode_row = np.flatnonzero(coms[:, 0] == stopcode_marker)
            startcode = int(coms[startcode_row[0], 1])
            stopcode = int(coms[stopcode_row[0], 1])
            s0, s1 = startcode - 1, stopcode
            entry["stim"] = stim[s0:s1]
            entry["sg"] = SG[s0:s1]
            entry["emgrms"] = RMS[s0:s1]
            rr0 = int(np.ceil(startcode / fcoef)) - 1
            rr1 = int(np.floor(stopcode / fcoef))
            entry["rr"] = RR[rr0:rr1]
        else:
            for marker in entry["codes"]:
                startind_rows = np.flatnonzero(coms[:, 0] == marker)
                for row in startind_rows:
                    startind1 = coms[row, 1]
                    candidates = np.flatnonzero((startinds + 1) >= startind1)
                    if candidates.size == 0:
                        continue
                    idx = candidates[0]
                    startcode = startinds[idx] + 1 - tbefore_samples
                    existing = entry["stim"]
                    N = existing.shape[1] if (existing.ndim == 2 and existing.shape[0] > 0) else 0
                    if N > 1:
                        stopcode = startcode + N - 1
                    else:
                        stopcode = stopinds[idx] + 1 - stimint_samples + tafter_samples

                    s0 = int(round(startcode)) - 1
                    s1 = int(round(stopcode))
                    seg_stim = stim[s0:s1]
                    seg_sg = SG[s0:s1]
                    seg_emgrms = RMS[s0:s1]
                    rr0 = int(np.ceil(startcode / fcoef)) - 1
                    rr1 = int(np.floor(stopcode / fcoef))
                    seg_rr = RR[rr0:rr1]

                    def _stack(existing_arr, new_row):
                        new_row = np.atleast_2d(new_row)
                        if existing_arr.size == 0:
                            return new_row
                        return np.vstack([np.atleast_2d(existing_arr), new_row])

                    entry["stim"] = _stack(entry["stim"], seg_stim)
                    entry["sg"] = _stack(entry["sg"], seg_sg)
                    entry["rr"] = _stack(entry["rr"], seg_rr)
                    entry["emgrms"] = _stack(entry["emgrms"], seg_emgrms)

    block1 = {name: {} for name in stim_names}
    block2 = {name: {} for name in stim_names}
    for name in stim_names:
        for key in ("stim", "sg", "rr", "emgrms"):
            arr = stimtype[name][key]
            if arr.ndim == 1:
                block1[name][key] = arr
                block2[name][key] = np.array([])
            else:
                block1[name][key] = arr[0, :] if arr.shape[0] >= 1 else np.array([])
                block2[name][key] = arr[1, :] if arr.shape[0] >= 2 else np.array([])


    # =============================================================================
    # 7. SG amplitude / force (baselines)
    # =============================================================================

    QC_AND_TARGET = BASELINE_CONDITIONS + TARGET_CONDITIONS


    def _max_avg(arr):
        if arr.size == 0:
            return None, None
        return float(np.max(arr)), float(np.mean(arr))


    sg_summary_rows = []
    for name in BASELINE_CONDITIONS:
        for block_label, block in (("Block 1", block1), ("Block 2", block2)):
            sg_arr = block[name]["sg"]
            max_amp, avg_amp = _max_avg(sg_arr)
            max_force = max_amp * calibration_factor if max_amp is not None else None
            avg_force = avg_amp * calibration_factor if avg_amp is not None else None
            sg_summary_rows.append({
                "Condition": name,
                "Block": block_label,
                "Role": "Baseline (QC only)",
                "Baseline_Value_g": None,
                "Baseline_Peaks_Excluded": None,
                "Baseline_Correction_Applied": False,
                "Max_SG_Amplitude_g": max_amp,
                "Avg_SG_Amplitude_g": avg_amp,
                "Max_Tongue_Force_N": max_force,
                "Avg_Tongue_Force_N": avg_force,
            })


    # =============================================================================
    # 8. Determine the REAL stimulus-delivery window within each target
    # =============================================================================

    def _real_stim_window(stim_segment, sample_rate, threshold=th):
        if stim_segment.size == 0:
            return None, None
        active = np.flatnonzero(stim_segment > threshold)
        if active.size == 0:
            return None, None
        onset_sec = active[0] / sample_rate
        offset_sec = active[-1] / sample_rate
        return onset_sec, offset_sec


    stim_windows = {name: {} for name in TARGET_CONDITIONS}
    for name in TARGET_CONDITIONS:
        for block_label, block in (("Block 1", block1), ("Block 2", block2)):
            onset, offset = _real_stim_window(block[name]["stim"], fs)
            stim_windows[name][block_label] = (onset, offset)


    # =============================================================================
    # 8b. Baseline correction -- ALWAYS applied, to every target condition/block.
    #     Baseline = median of the pre-stim window (the tbefore_sec seconds before
    #     the real stim onset) AFTER removing any peaks that fall in that window.
    #     Each block is corrected with its own pre-stim baseline.
    # =============================================================================


    def _prestim_baseline(sg_segment, onset_sec, sample_rate):
        out = {"value": None, "win_start_sec": None, "win_end_sec": None,
               "n_peaks_excluded": 0, "excluded_peak_locs": np.array([], dtype=int),
               "n_samples_used": 0, "n_samples_window": 0}
        if sg_segment.size == 0 or onset_sec is None:
            return out
        n_on = min(int(round(onset_sec * sample_rate)), sg_segment.size)
        n_start = max(0, n_on - int(round(tbefore_sec * sample_rate)))
        if n_on - n_start < 1:
            return out
        out["win_start_sec"] = n_start / sample_rate
        out["win_end_sec"] = n_on / sample_rate
        out["n_samples_window"] = n_on - n_start

        # Detect peaks on the WHOLE segment (so a peak straddling the window edge is
        # still found), then cut out every peak whose extent overlaps the window.
        peak_mask = np.zeros(sg_segment.size, dtype=bool)
        excluded_locs = []
        prom_threshold, _ = _auto_peak_params(sg_segment)
        if prom_threshold > 0:
            locs, _ = find_peaks(sg_segment,
                                 prominence=prom_threshold * BASELINE_PEAK_PROMINENCE_FRACTION)
            if locs.size:
                _, _, left_ips, right_ips = peak_widths(
                    sg_segment, locs, rel_height=BASELINE_PEAK_WIDTH_REL_HEIGHT)
                for pk, l, r in zip(locs, left_ips, right_ips):
                    lo = max(0, int(np.floor(l)))
                    hi = min(sg_segment.size, int(np.ceil(r)) + 1)
                    if hi > n_start and lo < n_on:
                        peak_mask[lo:hi] = True
                        excluded_locs.append(int(pk))

        prestim = sg_segment[n_start:n_on]
        keep = ~peak_mask[n_start:n_on]
        used = prestim[keep] if np.any(keep) else prestim  # fallback: whole window
        out["value"] = float(np.median(used))
        out["n_peaks_excluded"] = len(excluded_locs)
        out["excluded_peak_locs"] = np.array(excluded_locs, dtype=int)
        out["n_samples_used"] = int(used.size)
        return out

    baseline_info = {name: {} for name in TARGET_CONDITIONS}
    baseline_correction_rows = []
    for name in TARGET_CONDITIONS:
        for block_label, block in (("Block 1", block1), ("Block 2", block2)):
            onset, _ = stim_windows[name][block_label]
            info = _prestim_baseline(block[name]["sg"], onset, fs)
            corrected = info["value"] is not None
            if corrected:
                block[name]["sg"] = block[name]["sg"] - info["value"]
                print(f"  Baseline correction applied for {name} ({block_label}): "
                      f"subtracted {info['value']:.3f} g "
                      f"({info['n_peaks_excluded']} pre-stim peak(s) excluded)")
            info["corrected"] = corrected
            baseline_info[name][block_label] = info

            pct_used = (100.0 * info["n_samples_used"] / info["n_samples_window"]
                        if info["n_samples_window"] else None)
            baseline_correction_rows.append({
                "Condition": name,
                "Block": block_label,
                "PreStim_Window_Start_sec (rel. stim onset)": (
                    info["win_start_sec"] - onset if info["win_start_sec"] is not None else None),
                "PreStim_Window_End_sec (rel. stim onset)": (
                    info["win_end_sec"] - onset if info["win_end_sec"] is not None else None),
                "Baseline_Peaks_Excluded": info["n_peaks_excluded"],
                "PreStim_Samples_Used_pct": pct_used,
                "Baseline_Value_g": info["value"],
                "Correction_Applied": corrected,
            })
    baseline_correction_df = pd.DataFrame(baseline_correction_rows)


    # =============================================================================
    # 9. Swallow detection -- TARGET CONDITIONS ONLY, clipped to the real
    # =============================================================================

    def _filter_ringing_peaks(signal, locs, baseline_level, return_fraction=0.5):
        if locs.size < 2:
            return locs
        keep = np.ones(locs.size, dtype=bool)
        amps = signal[locs]
        for i in range(locs.size - 1):
            if not keep[i]:
                continue
            valley = signal[locs[i]:locs[i + 1] + 1].min()
            smaller_amp = min(amps[i], amps[i + 1])
            required = baseline_level + return_fraction * (smaller_amp - baseline_level)
            if valley > required:
                drop_idx = i if amps[i] < amps[i + 1] else i + 1
                keep[drop_idx] = False
        return locs[keep]


    def _detect_swallows_in_window(sg_segment, sample_rate, onset_sec, offset_sec):
        if sg_segment.size == 0 or onset_sec is None:
            return None
        prom_threshold, min_distance = _auto_peak_params(sg_segment)
        if prom_threshold <= 0:
            return None
        locs, props = find_peaks(sg_segment, prominence=prom_threshold, distance=min_distance)
        if locs.size == 0:
            return None
        baseline_level = float(np.percentile(sg_segment, 10))
        locs = _filter_ringing_peaks(sg_segment, locs, baseline_level)
        if locs.size == 0:
            return None
        times = locs / sample_rate
        keep = (times >= onset_sec) & (times <= offset_sec)
        if not np.any(keep):
            return None
        return {
            "locs": locs[keep],
            "times": times[keep],
            "amps": sg_segment[locs[keep]],
            "prom_threshold": prom_threshold,
        }


    swallow_events = {name: {} for name in TARGET_CONDITIONS}
    for name in TARGET_CONDITIONS:
        for block_label, block in (("Block 1", block1), ("Block 2", block2)):
            onset, offset = stim_windows[name][block_label]
            swallow_events[name][block_label] = _detect_swallows_in_window(
                block[name]["sg"], fs, onset, offset)


    for name in TARGET_CONDITIONS:
        for block_label in ("Block 1", "Block 2"):
            event = swallow_events[name][block_label]
            onset, offset = stim_windows[name][block_label]
            if event is None or onset is None:
                continue
            cutoff = onset + TIME_EXCLUDE
            counted_mask = event["times"] >= cutoff
            excluded_amps = event["amps"][~counted_mask]
            event["counted_mask"] = counted_mask
            event["max_peak_in_excluded_window_g"] = (
                float(np.max(excluded_amps)) if excluded_amps.size else None)


    def _selected_peak_mask(amps, n=NUMBER_PEAK_INCLUDE):
        mask = np.zeros(len(amps), dtype=bool)
        if len(amps) == 0:
            return mask
        if n == 0 or n >= len(amps):
            mask[:] = True
            return mask
        top_idx = np.argsort(amps)[-n:]
        mask[top_idx] = True
        return mask


    # =============================================================================
    # 9b. SG amplitude / force for TARGET CONDITIONS -- selected peaks + whole train:
    # =============================================================================

    for name in TARGET_CONDITIONS:
        for block_label, block in (("Block 1", block1), ("Block 2", block2)):
            event = swallow_events[name][block_label]
            if event is not None and event["amps"].size > 0:
                counted_mask = event.get("counted_mask", np.ones(event["amps"].size, dtype=bool))
                amps = event["amps"][counted_mask]
                if amps.size > 0:
                    selected_peak_mask = _selected_peak_mask(amps)
                    selected_amps = amps[selected_peak_mask]
                    max_amp = float(np.max(selected_amps))
                    avg_amp = float(np.mean(selected_amps))
                    max_force = max_amp * calibration_factor
                    avg_force = avg_amp * calibration_factor
                else:
                    max_amp = avg_amp = max_force = avg_force = None
            else:
                max_amp = avg_amp = max_force = avg_force = None

            onset, offset = stim_windows[name][block_label]
            sg_segment = block[name]["sg"]
            if sg_segment.size > 0 and onset is not None:
                s0 = int(round(onset * fs))
                s1 = int(round(offset * fs)) + 1
                whole_train = sg_segment[max(s0, 0):min(s1, sg_segment.size)]
            else:
                whole_train = np.array([])
            if whole_train.size > 0:
                max_amp_wt = float(np.max(whole_train))
                avg_amp_wt = float(np.mean(whole_train))
                max_force_wt = max_amp_wt * calibration_factor
                avg_force_wt = avg_amp_wt * calibration_factor
            else:
                max_amp_wt = avg_amp_wt = max_force_wt = avg_force_wt = None

            sg_summary_rows.append({
                "Condition": name,
                "Block": block_label,
                "Role": "Stimulated",
                "Baseline_Value_g": baseline_info[name][block_label]["value"],
                "Baseline_Peaks_Excluded": baseline_info[name][block_label]["n_peaks_excluded"],
                "Baseline_Correction_Applied": baseline_info[name][block_label]["corrected"],
                "Max_SG_Amplitude_g": max_amp,
                "Avg_SG_Amplitude_g": avg_amp,
                "Max_Tongue_Force_N": max_force,
                "Avg_Tongue_Force_N": avg_force,
                "Max_SG_Amplitude_WholeTrain_g": max_amp_wt,
                "Avg_SG_Amplitude_WholeTrain_g": avg_amp_wt,
                "Max_Tongue_Force_WholeTrain_N": max_force_wt,
                "Avg_Tongue_Force_WholeTrain_N": avg_force_wt,
            })
    sg_summary_df = pd.DataFrame(sg_summary_rows)


    def _n_counted(event):
        if event is None:
            return 0
        counted_mask = event.get("counted_mask")
        if counted_mask is None:
            return len(event["times"])
        return int(np.sum(counted_mask))


    def _draw_swallow_subplot(ax, sg_segment, sample_rate, event, title, window=None):
        onset = window[0] if (window is not None and window[0] is not None) else 0.0
        t = np.arange(1, len(sg_segment) + 1) / sample_rate - onset
        ax.plot(*_decimate_for_plot(t, sg_segment), color="tab:blue", linewidth=1.2)
        if window is not None and window[0] is not None:
            win_onset, win_offset = window
            ax.axvline(win_onset - onset, color="purple", linewidth=1.8, label="Stim window")
            ax.axvline(win_offset - onset, color="purple", linewidth=1.8)
            ax.axvline(TIME_EXCLUDE, color="orange", linewidth=1.6, linestyle="--",
                       label=f"Excluded first {TIME_EXCLUDE:g}s")
        if event is not None:
            shifted_times = event["times"] - onset
            counted_mask = event.get("counted_mask", np.ones(shifted_times.size, dtype=bool))
            counted_times = shifted_times[counted_mask]
            counted_amps = event["amps"][counted_mask]
            ax.plot(counted_times, counted_amps, linestyle=":", color="black",
                    linewidth=1.2, zorder=3)
            ax.plot(shifted_times, event["amps"], "o", color="green", markersize=7, zorder=4)
            selected_peak_mask = _selected_peak_mask(counted_amps)
            if selected_peak_mask.any():
                ax.scatter(counted_times[selected_peak_mask], counted_amps[selected_peak_mask],
                           facecolors="none", edgecolors="red",
                           linewidths=1.5, linestyle=":", s=170,
                           label=("All peaks (tongue force)" if NUMBER_PEAK_INCLUDE == 0 else f"Top {NUMBER_PEAK_INCLUDE} (tongue force)"), zorder=5)
            for k, (tt, aa) in enumerate(zip(counted_times, counted_amps), start=1):
                ax.annotate(str(k), (tt, aa), textcoords="offset points",
                            xytext=(6, 8), fontsize=11, color="black")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("SG Signal Amplitude (g)")
        ax.set_title(title)


    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))
    for ax, name in zip(axes1.ravel(), TARGET_CONDITIONS):
        event = swallow_events[name]["Block 1"]
        n = _n_counted(event)
        _draw_swallow_subplot(ax, block1[name]["sg"], fs, event,
                               f"SG Peaks - {name} (1st Stim), n={n}",
                               window=stim_windows[name]["Block 1"])
    fig1.suptitle("SG Signal Peaks - 1st Stimulus - Target Conditions")
    fig1.tight_layout()
    finish_figure(fig1, "swallow_peaks_block1")

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    for ax, name in zip(axes2.ravel(), TARGET_CONDITIONS):
        event = swallow_events[name]["Block 2"]
        n = _n_counted(event)
        _draw_swallow_subplot(ax, block2[name]["sg"], fs, event,
                               f"SG Peaks - {name} (2nd Stim), n={n}",
                               window=stim_windows[name]["Block 2"])
    fig2.suptitle("SG Signal Peaks - 2nd Stimulus - Target Conditions")
    fig2.tight_layout()
    finish_figure(fig2, "swallow_peaks_block2")

    for name in TARGET_CONDITIONS:
        for block_label, block in (("Block 1", block1), ("Block 2", block2)):
            event = swallow_events[name][block_label]
            n = _n_counted(event)
            fig_i, ax_i = plt.subplots(figsize=(7, 4.5))
            _draw_swallow_subplot(ax_i, block[name]["sg"], fs, event,
                                   f"SG Peaks - {name} ({block_label}), n={n}",
                                   window=stim_windows[name][block_label])
            fig_i.tight_layout()
            finish_figure(fig_i, f"swallow_individual_{name}_{block_label.replace(' ', '')}")


    # =============================================================================
    # 10. Swallow latency -- time from stim onset to the first counted
    # =============================================================================


    def _messy_signal_duration(latency_sec, window_sec=MESSY_SIGNAL_WINDOW_SEC):
        if latency_sec is None:
            return window_sec
        return min(latency_sec, window_sec)

    def _swallow_rate_hz(n_swallows, duration_sec):
        if duration_sec is None or n_swallows == 0:
            return None
        if duration_sec <= 0:
            return None
        return n_swallows / duration_sec

    def _inter_swallow_interval_ms(swallow_times):
        if swallow_times is None or len(swallow_times) < 2:
            return None
        return float(np.mean(np.diff(swallow_times)) * 1000.0)

    swallow_latency_rows = []
    for name in TARGET_CONDITIONS:
        for block_label, block in (("Block 1", block1), ("Block 2", block2)):
            event = swallow_events[name][block_label]
            onset, offset = stim_windows[name][block_label]
            n_swallows_full = len(event["times"]) if event else 0

            latency = None
            if event is not None and onset is not None and n_swallows_full > 0:
                latency = event["times"][0] - onset
            messy_duration = _messy_signal_duration(latency)
            messy_duration_uncapped_ms = (latency * 1000.0) if latency is not None else None

            counted_mask = event.get("counted_mask") if event is not None else None
            if event is not None and counted_mask is not None:
                counted_times = event["times"][counted_mask]
            else:
                counted_times = np.array([])
            n_swallows_counted = counted_times.size

            outcome_duration = None
            if onset is not None and offset is not None:
                outcome_duration = offset - (onset + TIME_EXCLUDE)
            swallow_rate = _swallow_rate_hz(n_swallows_counted, outcome_duration)
            isi_ms = _inter_swallow_interval_ms(counted_times)

            max_peak_excluded = event.get("max_peak_in_excluded_window_g") if event is not None else None

            swallow_latency_rows.append({
                "Condition": name,
                "Block": block_label,
                "N_swallows_in_stim_window": n_swallows_counted,
                "N_swallows_full_20sec": n_swallows_full,
                "Max_Peak_In_Excluded_Window_g": max_peak_excluded,
                "Swallow_latency_sec": latency,
                "Messy_Signal_Duration_sec": messy_duration,
                "Messy_Signal_Duration_Uncapped_ms": messy_duration_uncapped_ms,
                "Swallow_Rate_Hz": swallow_rate,
                "Inter_Swallow_Interval_ms": isi_ms,
            })
    swallow_latency_df = pd.DataFrame(swallow_latency_rows)


    # =============================================================================
    # 11. Maximum intensity -- among target conditions only
    # =============================================================================

    def _parse_intensity_uA(name):
        if len(name) >= 3 and name[0] in ("A", "D") and name[1:].isdigit():
            return int(name[1:])
        return None


    max_intensity_rows = []
    for block_label in ("Block 1", "Block 2"):
        best_name, best_uA, best_force = None, None, None
        for name in TARGET_CONDITIONS:
            row = [r for r in sg_summary_rows if r["Condition"] == name and r["Block"] == block_label][0]
            force = row["Max_Tongue_Force_N"]
            if force is not None and (best_force is None or force > best_force):
                best_name, best_uA, best_force = name, _parse_intensity_uA(name), force
        max_intensity_rows.append({
            "Block": block_label,
            "Condition_at_max_force": best_name,
            "Intensity_uA": best_uA,
            "Max_Tongue_Force_N": best_force,
        })
    max_intensity_df = pd.DataFrame(max_intensity_rows)


    # =============================================================================
    # 12. RR breath detection
    # =============================================================================


    def _detect_breaths(rr_signal, sample_rate, onset_sec):
        """Real breaths vs. tiny blips, judged against normal pre-stim breathing."""
        out = {"locs": np.array([], dtype=int), "times": np.array([]),
               "blip_locs": np.array([], dtype=int), "blip_times": np.array([]),
               "ref_prom": None, "prestim_ibi": None}
        if rr_signal.size < 3:
            return out

        def _reference(sig):
            rng = np.percentile(sig, 95) - np.percentile(sig, 5)
            if rng <= 0:
                return None, None
            pl, pp = find_peaks(sig, prominence=BREATH_REFERENCE_PROMINENCE_FRACTION * rng)
            if pl.size < 2:
                return None, None
            return float(np.median(pp["prominences"])), float(np.median(np.diff(pl)) / sample_rate)

        ref_prom = prestim_ibi = None
        if onset_sec is not None:
            n_on = min(int(round(onset_sec * sample_rate)), rr_signal.size)
            n_st = max(0, n_on - int(round(tbefore_sec * sample_rate)))
            if n_on - n_st > 3:
                ref_prom, prestim_ibi = _reference(rr_signal[n_st:n_on])
        if ref_prom is None:          # fallback: no usable pre-stim breathing
            ref_prom, prestim_ibi = _reference(rr_signal)
        if ref_prom is None:
            return out

        dist = max(1, int(round(BREATH_MIN_SPACING_FRACTION * prestim_ibi * sample_rate)))
        locs, props = find_peaks(rr_signal, prominence=BLIP_MIN_PROMINENCE_FRACTION * ref_prom,
                                 distance=dist)
        real = props["prominences"] >= BREATH_MIN_PROMINENCE_FRACTION * ref_prom
        out.update(locs=locs[real], times=locs[real] / sample_rate,
                   blip_locs=locs[~real], blip_times=locs[~real] / sample_rate,
                   ref_prom=ref_prom, prestim_ibi=prestim_ibi)
        return out


    def _local_normal_ibi(breath_locs, sample_rate, near_time_sec, window_sec=10.0):
        if breath_locs.size < 3:
            return None
        times = breath_locs / sample_rate
        mask = np.abs(times - near_time_sec) <= window_sec
        local_locs = breath_locs[mask]
        if local_locs.size < 3:
            ibi = np.diff(breath_locs) / sample_rate
        else:
            ibi = np.diff(local_locs) / sample_rate
        return float(np.median(ibi)) if ibi.size else None


    def _next_breath_after(breath_times, anchor_time_sec):
        later = breath_times[breath_times > anchor_time_sec]
        return float(later[0]) if later.size else None


    # =============================================================================
    # 13. Apnea -- computed two ways for every counted swallow:
    # =============================================================================


    swallow_rr_apnea_rows = []
    stim_rr_apnea_rows = []
    breath_events = {name: {} for name in TARGET_CONDITIONS}

    for name in TARGET_CONDITIONS:
        for block_label, block in (("Block 1", block1), ("Block 2", block2)):
            rr_segment = block[name]["rr"]
            onset, offset = stim_windows[name][block_label]
            breath_events[name][block_label] = _detect_breaths(rr_segment, fsRR, onset)
            breath_locs = breath_events[name][block_label]["locs"]
            breath_times = breath_events[name][block_label]["times"]
            breath_events[name][block_label]["lag_points"] = []   # (swallow time, lag ms) for plot stars
            event = swallow_events[name][block_label]

            if event is not None and breath_times.size:
                # Swallow-breath pairing within the RESPIRATORY CYCLE.
                # Each real breath's cycle runs from where the RR trace rises off
                # baseline to where it returns to baseline (within
                # BREATH_CYCLE_BASELINE_FRACTION of the breath's height); RR peak = T 0.
                # A swallow inside a cycle belongs to that breath (if inside two
                # overlapping cycles: the one with the nearer peak). Swallows outside
                # every cycle (flat baseline between breaths / apnea) get no lag.
                # One swallow per breath: if several swallows fall in the same cycle,
                # only the one closest to the RR peak keeps a lag.
                sw_times = np.asarray(event["times"], dtype=float)
                _, _, c_left, c_right = peak_widths(rr_segment, breath_locs,
                                                    rel_height=1.0 - BREATH_CYCLE_BASELINE_FRACTION)
                cyc_start, cyc_end = c_left / fsRR, c_right / fsRR
                cycle_idx = np.full(sw_times.size, -1)
                for j, t_sw in enumerate(sw_times):
                    inside = np.flatnonzero((cyc_start <= t_sw) & (t_sw <= cyc_end))
                    if inside.size:
                        cycle_idx[j] = inside[np.argmin(np.abs(breath_times[inside] - t_sw))]
                paired = np.zeros(sw_times.size, dtype=bool)
                for bi in np.unique(cycle_idx[cycle_idx >= 0]):
                    members = np.flatnonzero(cycle_idx == bi)
                    paired[members[np.argmin(np.abs(sw_times[members] - breath_times[bi]))]] = True
                for k, swallow_time in enumerate(event["times"], start=1):
                    next_breath = _next_breath_after(breath_times, swallow_time)
                    duration = (next_breath - swallow_time) if next_breath is not None else None
                    normal_ibi = _local_normal_ibi(breath_locs, fsRR, swallow_time)
                    verified = (duration is not None and normal_ibi is not None
                                and duration > VERIFIED_APNEA_MULTIPLIER * normal_ibi)
                    # Swallow-breath lag: swallow (SG peak) minus the RR peak (T = 0)
                    # of the respiratory cycle it falls in. Positive = after the peak
                    # (expiration); negative = before it (inspiration).
                    ci = int(cycle_idx[k - 1])
                    in_cycle = ci >= 0
                    is_paired = bool(paired[k - 1])
                    lag_ms = (swallow_time - breath_times[ci]) * 1000.0 if is_paired else None
                    swallow_rr_apnea_rows.append({
                        "Condition": name,
                        "Block": block_label,
                        "Swallow_number": k,
                        "Swallow_time_sec": swallow_time,
                        "Apnea_latency_sec (time from stim onset to swallow)": (
                            swallow_time - onset if onset is not None else None),
                        "Apnea_duration_sec (assumed, to next breath)": duration,
                        "Local_normal_IBI_sec": normal_ibi,
                        "Verified_apnea": verified,
                        "In_respiratory_cycle": in_cycle,
                        "Resp_cycle_start_sec": float(cyc_start[ci]) if in_cycle else None,
                        "Resp_cycle_RR_peak_time_sec": float(breath_times[ci]) if in_cycle else None,
                        "Resp_cycle_end_sec": float(cyc_end[ci]) if in_cycle else None,
                        "Swallow_RR_Lag_ms": lag_ms,
                        "Paired_with_breath": is_paired,
                        "Lag_sign": (None if lag_ms is None else
                                     "Positive" if lag_ms > 0 else "Negative" if lag_ms < 0 else "Zero"),
                        "Plot_star_color": (None if lag_ms is None else
                                            "Blue" if lag_ms > 0 else "Pink" if lag_ms < 0 else "Dark green"),
                        "In_excluded_window": (onset is not None and swallow_time < onset + TIME_EXCLUDE),
                        "During_extended_apnea": False,
                    })
                    if lag_ms is not None:
                        breath_events[name][block_label]["lag_points"].append((float(swallow_time), lag_ms))

            if onset is not None and breath_times.size:
                next_breath = _next_breath_after(breath_times, onset)
                duration = (next_breath - onset) if next_breath is not None else None
                normal_ibi = _local_normal_ibi(breath_locs, fsRR, onset)
                verified = (duration is not None and normal_ibi is not None
                            and duration > VERIFIED_APNEA_MULTIPLIER * normal_ibi)
                stim_rr_apnea_rows.append({
                    "Condition": name,
                    "Block": block_label,
                    "Apnea_latency_sec (time from stim onset; = 0 by definition)": 0.0,
                    "Apnea_duration_sec (assumed, stim onset to next breath)": duration,
                    "Local_normal_IBI_sec": normal_ibi,
                    "Verified_apnea": verified,
                })

    swallow_rr_apnea_df = pd.DataFrame(swallow_rr_apnea_rows)
    stim_rr_apnea_df = pd.DataFrame(stim_rr_apnea_rows)


    # Extended apnea = a stretch inside the stim window with no normal breathing,
    # lasting at least max(EXTENDED_APNEA_MIN_SEC, EXTENDED_APNEA_IBI_MULTIPLIER x the
    # normal pre-stim inter-breath interval). It ends when normal breathing resumes:
    # a real breath that has another real breath within that same gap. A single
    # isolated breath (or any blip) does NOT end an apnea.

    def _apnea_gap_threshold(prestim_ibi):
        if prestim_ibi is None:
            return EXTENDED_APNEA_MIN_SEC
        return max(EXTENDED_APNEA_MIN_SEC, EXTENDED_APNEA_IBI_MULTIPLIER * prestim_ibi)

    def _extended_apnea_episodes(breath_times, onset, offset, gap):
        if onset is None or offset is None:
            return []
        bt = np.sort(np.asarray(breath_times, dtype=float))
        if bt.size >= 2:
            d = np.diff(bt)
            normal = bt[np.r_[False, d < gap] | np.r_[d < gap, False]]
        else:
            normal = np.array([])
        pts = np.r_[-np.inf, normal, np.inf]
        episodes = []
        for a, b in zip(pts[:-1], pts[1:]):
            start, end = max(a, onset), min(b, offset)
            if end - start >= gap:
                episodes.append({"start": float(start), "end": float(end),
                                 "ended_by": "Stim offset" if b > offset else "Breathing resumed"})
        return episodes

    def _swallows_during_apnea(swallow_times, episode_start, episode_end):
        if swallow_times is None or episode_start is None or episode_end is None:
            return 0
        return int(np.sum((swallow_times >= episode_start) & (swallow_times <= episode_end)))

    apnea_episode_rows = []
    apnea_summary_rows = []
    for name in TARGET_CONDITIONS:
        for block_label in ("Block 1", "Block 2"):
            sub = swallow_rr_apnea_df[
                (swallow_rr_apnea_df["Condition"] == name) & (swallow_rr_apnea_df["Block"] == block_label)
            ] if not swallow_rr_apnea_df.empty else pd.DataFrame()
            n_assumed = len(sub)
            n_verified = int(sub["Verified_apnea"].sum()) if len(sub) else 0
            first_latency = (sub["Apnea_latency_sec (time from stim onset to swallow)"].iloc[0]
                              if len(sub) else None)
            mean_duration = (sub["Apnea_duration_sec (assumed, to next breath)"].mean()
                              if len(sub) else None)

            stim_row = stim_rr_apnea_df[
                (stim_rr_apnea_df["Condition"] == name) & (stim_rr_apnea_df["Block"] == block_label)
            ] if not stim_rr_apnea_df.empty else pd.DataFrame()
            stim_duration = (stim_row["Apnea_duration_sec (assumed, stim onset to next breath)"].iloc[0]
                              if len(stim_row) else None)
            stim_verified = bool(stim_row["Verified_apnea"].iloc[0]) if len(stim_row) else False

            window_onset, window_offset = stim_windows[name][block_label]
            breaths = breath_events[name][block_label]
            gap = _apnea_gap_threshold(breaths["prestim_ibi"])
            episodes = _extended_apnea_episodes(breaths["times"], window_onset, window_offset, gap)
            breaths["episodes"] = episodes

            event = swallow_events[name][block_label]
            full_swallow_times = event["times"] if event is not None else None

            total_apnea = 0.0
            total_swallows = 0
            for k, ep in enumerate(episodes, start=1):
                dur = ep["end"] - ep["start"]
                n_sw = _swallows_during_apnea(full_swallow_times, ep["start"], ep["end"])
                total_apnea += dur
                total_swallows += n_sw
                apnea_episode_rows.append({
                    "Condition": name,
                    "Block": block_label,
                    "Episode": k,
                    "Start_sec (rel. stim onset)": ep["start"] - window_onset,
                    "End_sec (rel. stim onset)": ep["end"] - window_onset,
                    "Duration_sec": dur,
                    "Ended_by": ep["ended_by"],
                    "N_Swallows": n_sw,
                    "Swallow_Rate_Hz": n_sw / dur if dur > 0 else None,
                })

            if episodes:
                longest = max(episodes, key=lambda e: e["end"] - e["start"])
                extended_apnea = longest["end"] - longest["start"]
                episode_start = longest["start"] - window_onset
                episode_end = longest["end"] - window_onset
            else:
                extended_apnea = episode_start = episode_end = None

            stim_dur = (window_offset - window_onset) if window_onset is not None else None
            n_blips = (int(np.sum((breaths["blip_times"] >= window_onset)
                                  & (breaths["blip_times"] <= window_offset)))
                       if window_onset is not None else 0)
            n_swallows_during_apnea = total_swallows

            # Mark this block's swallows that fall inside an extended apnea episode
            if len(sub):
                in_ep = np.zeros(len(sub), dtype=bool)
                st = sub["Swallow_time_sec"].to_numpy()
                for ep in episodes:
                    in_ep |= (st >= ep["start"]) & (st <= ep["end"])
                swallow_rr_apnea_df.loc[sub.index, "During_extended_apnea"] = in_ep

            # Swallow-breath lag summary (all detected swallows in the block)
            lags = sub["Swallow_RR_Lag_ms"].dropna().to_numpy(dtype=float) if len(sub) else np.array([])
            pos_vals, neg_vals = lags[lags > 0], lags[lags < 0]
            _fmt = lambda v: ", ".join(f"{x:.1f}" for x in v) if v.size else None
            n_lag = int(lags.size)
            n_pos, n_neg = int(np.sum(lags > 0)), int(np.sum(lags < 0))

            apnea_summary_rows.append({
                "Condition": name,
                "Block": block_label,
                f"{APNEA_LABEL_SWALLOW_RR}_rate_assumed": n_assumed,
                f"{APNEA_LABEL_SWALLOW_RR}_rate_verified": n_verified,
                f"{APNEA_LABEL_SWALLOW_RR}_first_latency_sec": first_latency,
                f"{APNEA_LABEL_SWALLOW_RR}_mean_duration_sec": mean_duration,
                f"{APNEA_LABEL_STIM_RR}_duration_sec": stim_duration,
                f"{APNEA_LABEL_STIM_RR}_verified": stim_verified,
                "Extended_Apnea_Duration_sec": extended_apnea,
                "Extended_Apnea_Start_sec": episode_start,
                "Extended_Apnea_End_sec": episode_end,
                "N_Swallows_During_Apnea": n_swallows_during_apnea,
                "N_Extended_Apnea_Episodes": len(episodes),
                "Total_Apnea_Duration_sec": total_apnea if window_onset is not None else None,
                "Stim_Duration_sec": stim_dur,
                "Breathing_Duration_sec": (stim_dur - total_apnea) if stim_dur is not None else None,
                "Swallow_Rate_During_Apnea_Hz": (total_swallows / total_apnea) if total_apnea > 0 else None,
                "Normal_IBI_PreStim_sec": breaths["prestim_ibi"],
                "Apnea_Gap_Threshold_sec": gap,
                "N_Blips_Rejected_in_Stim": n_blips,
                "N_Swallows_Lag_Measured": n_lag,
                "Swallow_RR_Lag_mean_ms": float(np.mean(lags)) if n_lag else None,
                "Swallow_RR_Lag_median_ms": float(np.median(lags)) if n_lag else None,
                "N_Lag_Positive": n_pos,
                "Pct_Lag_Positive": 100.0 * n_pos / n_lag if n_lag else None,
                "N_Lag_Negative": n_neg,
                "Pct_Lag_Negative": 100.0 * n_neg / n_lag if n_lag else None,
                "N_Lag_Zero": n_lag - n_pos - n_neg,
                "Lag_Positive_values_ms": _fmt(pos_vals),
                "Lag_Positive_mean_ms": float(np.mean(pos_vals)) if pos_vals.size else None,
                "Lag_Positive_median_ms": float(np.median(pos_vals)) if pos_vals.size else None,
                "Lag_Negative_values_ms": _fmt(neg_vals),
                "Lag_Negative_mean_ms": float(np.mean(neg_vals)) if neg_vals.size else None,
                "Lag_Negative_median_ms": float(np.median(neg_vals)) if neg_vals.size else None,
                "N_Swallows_Outside_Resp_Cycle": (int((~sub["In_respiratory_cycle"].astype(bool)).sum())
                                                  if len(sub) else 0),
            })
    apnea_summary_df = pd.DataFrame(apnea_summary_rows)
    # Swallow-Breathing Coordination (SBC) results get their own sheets
    sbc_summary_df = apnea_summary_df[["Condition", "Block"] + SBC_SUMMARY_COLS].copy()
    apnea_summary_df = apnea_summary_df.drop(columns=SBC_SUMMARY_COLS)
    sbc_value_cols = ["Condition", "Block", "Swallow_number", "Swallow_time_sec", "Resp_cycle_RR_peak_time_sec",
                      "SBC_Positive_ms", "SBC_Negative_ms", "SBC_Zero", "Plot_star_color",
                      "In_excluded_window", "During_extended_apnea"]
    if not swallow_rr_apnea_df.empty and "Swallow_RR_Lag_ms" in swallow_rr_apnea_df.columns:
        _pr = swallow_rr_apnea_df[swallow_rr_apnea_df["Swallow_RR_Lag_ms"].notna()].copy()
        _lag = _pr["Swallow_RR_Lag_ms"].astype(float)
        _pr["SBC_Positive_ms"] = _lag.where(_lag > 0)
        _pr["SBC_Negative_ms"] = _lag.where(_lag < 0)
        _pr["SBC_Zero"] = _lag == 0
        sbc_values_df = _pr[sbc_value_cols].reset_index(drop=True)
    else:
        sbc_values_df = pd.DataFrame(columns=sbc_value_cols)
    apnea_episodes_df = pd.DataFrame(apnea_episode_rows, columns=[
        "Condition", "Block", "Episode", "Start_sec (rel. stim onset)", "End_sec (rel. stim onset)",
        "Duration_sec", "Ended_by", "N_Swallows", "Swallow_Rate_Hz"])


    # =============================================================================
    # 13a. Excluded window -- the first TIME_EXCLUDE seconds after stim onset.
    #      Same calculations as the counted part of the block, applied only to
    #      peaks/signal inside [onset, onset + TIME_EXCLUDE).
    # =============================================================================

    excluded_window_rows = []
    for name in TARGET_CONDITIONS:
        for block_label, block in (("Block 1", block1), ("Block 2", block2)):
            onset, offset = stim_windows[name][block_label]
            event = swallow_events[name][block_label]
            sg_segment = block[name]["sg"]

            if onset is not None:
                win_end = min(onset + TIME_EXCLUDE, offset)
                win_duration = win_end - onset
            else:
                win_end, win_duration = None, None

            # Peaks in the excluded window (= detected peaks that are NOT counted)
            if event is not None and "counted_mask" in event:
                ex_mask = ~event["counted_mask"]
                ex_times = event["times"][ex_mask]
                ex_amps = event["amps"][ex_mask]
            else:
                ex_times, ex_amps = np.array([]), np.array([])
            n_peaks = int(ex_times.size)

            # Peak amplitude / force (same top-N rule as the counted block)
            if n_peaks > 0:
                sel = _selected_peak_mask(ex_amps)
                max_amp = float(np.max(ex_amps[sel]))
                avg_amp = float(np.mean(ex_amps[sel]))
                max_force, avg_force = max_amp * calibration_factor, avg_amp * calibration_factor
            else:
                max_amp = avg_amp = max_force = avg_force = None

            # Whole-window amplitude / force (all samples in the excluded window)
            if sg_segment.size > 0 and onset is not None and win_duration > 0:
                s0 = max(int(round(onset * fs)), 0)
                s1 = min(int(round(win_end * fs)), sg_segment.size)
                ww = sg_segment[s0:s1]
            else:
                ww = np.array([])
            if ww.size > 0:
                max_amp_ww, avg_amp_ww = float(np.max(ww)), float(np.mean(ww))
                max_force_ww = max_amp_ww * calibration_factor
                avg_force_ww = avg_amp_ww * calibration_factor
            else:
                max_amp_ww = avg_amp_ww = max_force_ww = avg_force_ww = None

            latency = float(ex_times[0] - onset) if n_peaks > 0 else None
            rate = _swallow_rate_hz(n_peaks, win_duration)
            isi_ms = _inter_swallow_interval_ms(ex_times)

            # Breaths and Swallow-RR apnea for swallows in this window
            breaths = breath_events[name][block_label]
            if onset is not None and breaths["times"].size:
                n_breaths = int(np.sum((breaths["times"] >= onset) & (breaths["times"] < win_end)))
            else:
                n_breaths = 0
            if not swallow_rr_apnea_df.empty and onset is not None:
                sub = swallow_rr_apnea_df[
                    (swallow_rr_apnea_df["Condition"] == name)
                    & (swallow_rr_apnea_df["Block"] == block_label)
                    & (swallow_rr_apnea_df["Swallow_time_sec"] >= onset)
                    & (swallow_rr_apnea_df["Swallow_time_sec"] < win_end)
                ]
            else:
                sub = pd.DataFrame()
            ap_n_assumed = len(sub)
            ap_n_verified = int(sub["Verified_apnea"].sum()) if len(sub) else 0
            ap_first_latency = (float(sub["Apnea_latency_sec (time from stim onset to swallow)"].iloc[0])
                                if len(sub) else None)
            ap_mean_duration = (float(sub["Apnea_duration_sec (assumed, to next breath)"].mean())
                                if len(sub) else None)

            excluded_window_rows.append({
                "Condition": name,
                "Block": block_label,
                "Excluded_Window_Duration_sec": win_duration,
                "N_peaks_in_excluded_window": n_peaks,
                "Max_SG_Amplitude_g": max_amp,
                "Avg_SG_Amplitude_g": avg_amp,
                "Max_Tongue_Force_N": max_force,
                "Avg_Tongue_Force_N": avg_force,
                "Max_SG_Amplitude_WholeWindow_g": max_amp_ww,
                "Avg_SG_Amplitude_WholeWindow_g": avg_amp_ww,
                "Max_Tongue_Force_WholeWindow_N": max_force_ww,
                "Avg_Tongue_Force_WholeWindow_N": avg_force_ww,
                "First_Peak_latency_sec": latency,
                "Swallow_Rate_Hz": rate,
                "Inter_Swallow_Interval_ms": isi_ms,
                "N_breaths_in_excluded_window": n_breaths,
                f"{APNEA_LABEL_SWALLOW_RR}_rate_assumed": ap_n_assumed,
                f"{APNEA_LABEL_SWALLOW_RR}_rate_verified": ap_n_verified,
                f"{APNEA_LABEL_SWALLOW_RR}_first_latency_sec": ap_first_latency,
                f"{APNEA_LABEL_SWALLOW_RR}_mean_duration_sec": ap_mean_duration,
            })
    excluded_window_df = pd.DataFrame(excluded_window_rows)


    # =============================================================================
    # 13b. RR breath-detection plots -- same approach as the SG swallow plots:
    # =============================================================================

    BREATH_MARKER = dict(marker="o", linestyle="none", color="blue",
                         markeredgecolor="black", markeredgewidth=0.6, markersize=6, zorder=4)

    def _draw_rr_apnea_overlays(ax, rr_segment, breaths, window, onset):
        """Grey crosses on rejected blips + dark-green lines at extended-apnea boundaries."""
        if breaths is None or window is None or window[0] is None:
            return
        win_onset, win_offset = window
        bl = breaths["blip_locs"]
        if bl.size:
            bt = breaths["blip_times"]
            m = (bt >= win_onset) & (bt <= win_offset)
            if np.any(m):
                ax.plot(bt[m] - onset, rr_segment[bl[m]], "x", color="grey", markersize=9,
                        mew=2, zorder=5, label=f"Blip, not a breath ({int(m.sum())})")
        episodes = breaths.get("episodes", [])
        for k, ep in enumerate(episodes, start=1):
            for edge in (ep["start"], ep["end"]):
                ax.axvline(edge - onset, color="darkgreen", linewidth=2.0, zorder=3,
                           label="Extended apnea boundary" if (k == 1 and edge == ep["start"]) else None)
            ax.annotate(f"Apnea {k}\n{ep['end'] - ep['start']:.2f} s",
                        xy=((ep["start"] + ep["end"]) / 2 - onset, 0.92),
                        xycoords=("data", "axes fraction"), ha="center", va="top",
                        fontsize=8, color="darkgreen", fontweight="bold")
        # Swallow-breath lag stars, at the top of each paired swallow's dashed line:
        # blue = positive lag (after the RR peak), pink = negative (before it),
        # dark green = zero (swallow at the same time as the RR peak).
        lag_pts = breaths.get("lag_points", [])
        star_groups = (("Positive lag (after RR peak)", "blue", lambda v: v > 0),
                       ("Negative lag (before RR peak)", "hotpink", lambda v: v < 0),
                       ("Zero lag (same time as RR peak)", "darkgreen", lambda v: v == 0))
        for label, colr, test in star_groups:
            xs = [ts - onset for ts, lag in lag_pts if test(lag)]
            if xs:
                ax.plot(xs, [0.985] * len(xs), "*", color=colr, markersize=11, markeredgecolor="black",
                        markeredgewidth=0.5, transform=ax.get_xaxis_transform(), clip_on=False, zorder=6,
                        linestyle="none", label=f"{label} ({len(xs)})")
        if bl.size or episodes or lag_pts:
            ax.legend(loc="lower right", fontsize=7, framealpha=0.85)

    def _apnea_title_suffix(breaths):
        episodes = breaths.get("episodes", []) if breaths is not None else []
        total = sum(ep["end"] - ep["start"] for ep in episodes)
        return f", {len(episodes)} extended apnea ({total:.2f} s)"


    def _draw_breath_subplot(ax, rr_segment, sample_rate, breaths, swallow_times, title, window=None):
        onset = window[0] if (window is not None and window[0] is not None) else 0.0
        t = np.arange(1, len(rr_segment) + 1) / sample_rate - onset
        ax.plot(*_decimate_for_plot(t, rr_segment), color="tab:blue", linewidth=1.0)
        if window is not None and window[0] is not None:
            win_onset, win_offset = window
            ax.axvline(win_onset - onset, color="purple", linewidth=1.8, label="Stim window")
            ax.axvline(win_offset - onset, color="purple", linewidth=1.8)
        if breaths is not None and breaths["locs"].size:
            breath_amps = rr_segment[breaths["locs"]]
            breath_times = breaths["times"]
            if window is not None and window[0] is not None:
                win_onset, win_offset = window
                in_window = (breath_times >= win_onset) & (breath_times <= win_offset)
            else:
                in_window = np.ones(breath_times.size, dtype=bool)
            shifted_breath_times = breath_times[in_window] - onset
            ax.plot(shifted_breath_times, breath_amps[in_window], **BREATH_MARKER)
            for k, (tt, aa) in enumerate(zip(shifted_breath_times, breath_amps[in_window]), start=1):
                ax.annotate(str(k), (tt, aa), textcoords="offset points",
                            xytext=(5, 7), fontsize=9, color="black")
        if swallow_times is not None:
            for st in swallow_times:
                ax.axvline(st - onset, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        _draw_rr_apnea_overlays(ax, rr_segment, breaths, window, onset)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("RR Signal Amplitude (V)")
        ax.set_title(title)


    def _n_breaths_in_window(breaths, window):
        if breaths is None or breaths["locs"].size == 0:
            return 0
        if window is None or window[0] is None:
            return breaths["locs"].size
        onset, offset = window
        return int(np.sum((breaths["times"] >= onset) & (breaths["times"] <= offset)))


    fig1r, axes1r = plt.subplots(2, 2, figsize=(12, 8))
    for ax, name in zip(axes1r.ravel(), TARGET_CONDITIONS):
        breaths = breath_events[name]["Block 1"]
        event = swallow_events[name]["Block 1"]
        swallow_times = event["times"] if event else []
        n_swallows = len(swallow_times)
        window = stim_windows[name]["Block 1"]
        n = _n_breaths_in_window(breaths, window)
        _draw_breath_subplot(ax, block1[name]["rr"], fsRR, breaths, swallow_times,
                              f"RR Breaths - {name} (1st Stim), n={n} breaths, {n_swallows} swallows (SG){_apnea_title_suffix(breaths)}\n(red dashed = swallow onset)",
                              window=window)
    fig1r.suptitle("RR Breath Detection - 1st Stimulus - Target Conditions")
    fig1r.tight_layout()
    finish_figure(fig1r, "rr_breaths_block1")

    fig2r, axes2r = plt.subplots(2, 2, figsize=(12, 8))
    for ax, name in zip(axes2r.ravel(), TARGET_CONDITIONS):
        breaths = breath_events[name]["Block 2"]
        event = swallow_events[name]["Block 2"]
        swallow_times = event["times"] if event else []
        n_swallows = len(swallow_times)
        window = stim_windows[name]["Block 2"]
        n = _n_breaths_in_window(breaths, window)
        _draw_breath_subplot(ax, block2[name]["rr"], fsRR, breaths, swallow_times,
                              f"RR Breaths - {name} (2nd Stim), n={n} breaths, {n_swallows} swallows (SG){_apnea_title_suffix(breaths)}\n(red dashed = swallow onset)",
                              window=window)
    fig2r.suptitle("RR Breath Detection - 2nd Stimulus - Target Conditions")
    fig2r.tight_layout()
    finish_figure(fig2r, "rr_breaths_block2")

    for name in TARGET_CONDITIONS:
        for block_label, block in (("Block 1", block1), ("Block 2", block2)):
            breaths = breath_events[name][block_label]
            event = swallow_events[name][block_label]
            swallow_times = event["times"] if event else []
            n_swallows = len(swallow_times)
            window = stim_windows[name][block_label]
            n = _n_breaths_in_window(breaths, window)
            fig_ir, ax_ir = plt.subplots(figsize=(8, 4.5))
            _draw_breath_subplot(ax_ir, block[name]["rr"], fsRR, breaths, swallow_times,
                                  f"RR Breaths - {name} ({block_label}), n={n} breaths, {n_swallows} swallows (SG){_apnea_title_suffix(breaths)}\n(red dashed = swallow onset)",
                                  window=window)
            fig_ir.tight_layout()
            finish_figure(fig_ir, f"rr_breaths_individual_{name}_{block_label.replace(' ', '')}")


    # =============================================================================
    # 13c. Combined SG + RR summary figures
    # =============================================================================

    SUMMARY_FIGURE_ORDER = ["A40", "A30", "D40", "D30"]
    BLOCK_PLOT_COLOR = {"Block 1": "tab:blue", "Block 2": "tab:red"}


    def _draw_sg_single(ax, name, block_label, block):
        seg = block[name]["sg"]
        color = BLOCK_PLOT_COLOR[block_label]
        window = stim_windows[name][block_label]
        onset = window[0] if window[0] is not None else 0.0
        t = np.arange(1, len(seg) + 1) / fs - onset
        ax.plot(*_decimate_for_plot(t, seg), color=color, linewidth=1.1)
        if window[0] is not None:
            ax.axvline(window[0] - onset, color="purple", linewidth=1.8)
            ax.axvline(window[1] - onset, color="purple", linewidth=1.8)
            ax.axvline(TIME_EXCLUDE, color="orange", linewidth=1.4, linestyle="--")
        event = swallow_events[name][block_label]
        n = _n_counted(event)
        if event is not None:
            shifted_times = event["times"] - onset
            counted_mask = event.get("counted_mask", np.ones(shifted_times.size, dtype=bool))
            counted_times = shifted_times[counted_mask]
            counted_amps = event["amps"][counted_mask]
            ax.plot(counted_times, counted_amps, linestyle=":", color="black",
                    linewidth=1.2, zorder=3)
            ax.plot(shifted_times, event["amps"], "o", color="green", markersize=7, zorder=4)
            selected_peak_mask = _selected_peak_mask(counted_amps)
            if selected_peak_mask.any():
                ax.scatter(counted_times[selected_peak_mask], counted_amps[selected_peak_mask],
                           facecolors="none", edgecolors="red",
                           linewidths=1.5, linestyle=":", s=170,
                           label=("All peaks (tongue force)" if NUMBER_PEAK_INCLUDE == 0 else f"Top {NUMBER_PEAK_INCLUDE} (tongue force)"), zorder=5)
            for k, (tt, aa) in enumerate(zip(counted_times, counted_amps), start=1):
                ax.annotate(str(k), (tt, aa), textcoords="offset points",
                            xytext=(6, 8), fontsize=10, color="black")

        # --- Baseline validation ---
        # The trace is baseline-corrected, so the baseline sits at 0 g on the left
        # axis. The right axis shows the RAW (uncorrected) signal, where the same
        # dotted line reads the Baseline_Value_g reported in the Excel sheets.
        binfo = baseline_info[name][block_label]
        if binfo["value"] is not None:
            bval = binfo["value"]
            ax.axvspan(binfo["win_start_sec"] - onset, binfo["win_end_sec"] - onset,
                       color="grey", alpha=0.12, zorder=0, label="Baseline window")
            ax.axhline(0.0, color="black", linestyle=":", linewidth=1.6, zorder=2,
                       label=f"Baseline = {bval:.3f} g (raw)")
            if binfo["excluded_peak_locs"].size:
                ex_locs = binfo["excluded_peak_locs"]
                ax.plot(ex_locs / fs - onset, seg[ex_locs], "x", color="grey",
                        markersize=9, mew=2, zorder=4,
                        label=f"Excluded from baseline ({ex_locs.size})")
            sec_ax = ax.secondary_yaxis("right", functions=(lambda y, b=bval: y + b,
                                                            lambda y, b=bval: y - b))
            sec_ax.set_ylabel("Raw SG (g, before correction)")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("SG Amplitude, baseline-corrected (g)")
        ax.set_title(f"SG Swallow Peaks - {name} ({block_label}), n={n}")


    def _draw_rr_single(ax, name, block_label, block):
        seg = block[name]["rr"]
        color = BLOCK_PLOT_COLOR[block_label]
        window = stim_windows[name][block_label]
        onset = window[0] if window[0] is not None else 0.0
        t = np.arange(1, len(seg) + 1) / fsRR - onset
        ax.plot(*_decimate_for_plot(t, seg), color=color, linewidth=1.0)
        if window[0] is not None:
            ax.axvline(window[0] - onset, color="purple", linewidth=1.8)
            ax.axvline(window[1] - onset, color="purple", linewidth=1.8)
        breaths = breath_events[name][block_label]
        n_breaths = _n_breaths_in_window(breaths, window)
        if breaths is not None and breaths["locs"].size:
            breath_amps = seg[breaths["locs"]]
            in_win = ((breaths["times"] >= window[0]) & (breaths["times"] <= window[1])
                      if window[0] is not None else np.ones(breaths["times"].size, dtype=bool))
            shifted_breath_times = breaths["times"][in_win] - onset
            ax.plot(shifted_breath_times, breath_amps[in_win], **BREATH_MARKER)
        event = swallow_events[name][block_label]
        n_swallows = len(event["times"]) if event else 0
        if event is not None:
            for st in event["times"]:
                ax.axvline(st - onset, color="red", linestyle="--", linewidth=0.7, alpha=0.6)
        _draw_rr_apnea_overlays(ax, seg, breaths, window, onset)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("RR Signal Amplitude (V)")
        ax.set_title(f"RR Breaths - {name} ({block_label}), n={n_breaths} breaths, {n_swallows} swallows (SG)"
                     f"{_apnea_title_suffix(breaths)}\n"
                     f"(red dashed = swallow onset)")


    for name in SUMMARY_FIGURE_ORDER:
        for block_label, block in (("Block 1", block1), ("Block 2", block2)):
            fig_s, (ax_sg, ax_rr) = plt.subplots(2, 1, figsize=(10, 9))
            _draw_sg_single(ax_sg, name, block_label, block)
            _draw_rr_single(ax_rr, name, block_label, block)
            color_word = "Blue" if block_label == "Block 1" else "Red"
            fig_s.suptitle(f"{name} - Combined SG + RR Summary ({block_label}, {color_word})", fontsize=13)
            fig_s.tight_layout()
            finish_figure(fig_s, f"summary_combined_{name}_{block_label.replace(' ', '')}")


    # =============================================================================
    # 14. 4-panel diagnostic figures (Stim / EMG RMS / RR / SG)
    # =============================================================================

    BLOCK_COLOR = {"Block 1": ("tab:blue", "Blue"), "Block 2": ("tab:red", "Red")}

    for name in TARGET_CONDITIONS:
        for block_label, block in (("Block 1", block1), ("Block 2", block2)):
            color, color_word = BLOCK_COLOR[block_label]
            stim_seg = block[name]["stim"]
            emgrms_seg = block[name]["emgrms"]
            rr_seg = block[name]["rr"]
            sg_seg = block[name]["sg"]
            if stim_seg.size == 0:
                continue

            fig, axes = plt.subplots(4, 1, figsize=(9, 10))
            ordinal = "1st" if block_label == "Block 1" else "2nd"

            panels = [
                (axes[0], stim_seg, f"Stim - {name} ({ordinal} Stim)", "1.0 unit = 5 seconds"),
                (axes[1], emgrms_seg, f"EMG RMS - {name} ({ordinal} Stim)", "1.0 unit = 5 seconds"),
                (axes[2], rr_seg, f"RR - {name} ({ordinal} Stim)", "0.5 unit = 5 seconds"),
                (axes[3], sg_seg, f"SG - {name} ({ordinal} Stim)", "1.0 unit = 5 seconds"),
            ]
            for ax, seg, title, xlabel in panels:
                idx = np.arange(1, len(seg) + 1)
                ax.plot(*_decimate_for_plot(idx, seg), color=color, linewidth=0.8)
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.set_xlabel(xlabel)

            fig.suptitle(f"{name} Stimulus - {ordinal} Stimulus ({color_word})", fontsize=14)
            fig.tight_layout()
            finish_figure(fig, f"4panel_{name}_{block_label.replace(' ', '')}")


    # =============================================================================
    # 15. Excel export -- SG + Respiratory only, target conditions + baseline QC
    # =============================================================================

    UNIT_RENAME_MAP = {
        "Max_SG_Amplitude_g": "Max_SG_Amplitude (g)",
        "Avg_SG_Amplitude_g": "Avg_SG_Amplitude (g)",
        "Max_Tongue_Force_N": "Max_Tongue_Force (N)",
        "Avg_Tongue_Force_N": "Avg_Tongue_Force (N)",
        "Max_SG_Amplitude_WholeTrain_g": "Max_SG_Amplitude_WholeTrain (g)",
        "Avg_SG_Amplitude_WholeTrain_g": "Avg_SG_Amplitude_WholeTrain (g)",
        "Max_Tongue_Force_WholeTrain_N": "Max_Tongue_Force_WholeTrain (N)",
        "Avg_Tongue_Force_WholeTrain_N": "Avg_Tongue_Force_WholeTrain (N)",
        "N_swallows_in_stim_window": "N_swallows_in_stim_window (#)",
        "N_swallows_full_20sec": "N_swallows_full_20sec (#)",
        "Max_Peak_In_Excluded_Window_g": "Max_Peak_In_Excluded_Window (g)",
        "Messy_Signal_Duration_Uncapped_ms": "Messy_Signal_Duration_Uncapped (ms)",
        "Swallow_latency_sec": "Swallow_latency (s)",
        "Messy_Signal_Duration_sec": "Messy_Signal_Duration (s)",
        "Swallow_Rate_Hz": "Swallow_Rate (Hz)",
        "Inter_Swallow_Interval_ms": "Inter_Swallow_Interval (ms)",
        "Intensity_uA": "Intensity (uA)",
        "Swallow_time_sec": "Swallow_time (s)",
        "Local_normal_IBI_sec": "Local_normal_IBI (s)",
        "Apnea_latency_sec (time from stim onset to swallow)": "Apnea_latency (s) (time from stim onset to swallow)",
        "Apnea_duration_sec (assumed, to next breath)": "Apnea_duration (s) (assumed, to next breath)",
        "Apnea_latency_sec (time from stim onset; = 0 by definition)": "Apnea_latency (s) (time from stim onset; = 0 by definition)",
        "Apnea_duration_sec (assumed, stim onset to next breath)": "Apnea_duration (s) (assumed, stim onset to next breath)",
        f"{APNEA_LABEL_SWALLOW_RR}_rate_assumed": f"{APNEA_LABEL_SWALLOW_RR}_rate_assumed (#)",
        f"{APNEA_LABEL_SWALLOW_RR}_rate_verified": f"{APNEA_LABEL_SWALLOW_RR}_rate_verified (#)",
        f"{APNEA_LABEL_SWALLOW_RR}_first_latency_sec": f"{APNEA_LABEL_SWALLOW_RR}_first_latency (s)",
        f"{APNEA_LABEL_SWALLOW_RR}_mean_duration_sec": f"{APNEA_LABEL_SWALLOW_RR}_mean_duration (s)",
        f"{APNEA_LABEL_STIM_RR}_duration_sec": f"{APNEA_LABEL_STIM_RR}_duration (s)",
        "Extended_Apnea_Duration_sec": "Extended_Apnea_Duration (s) (longest episode)",
        "Extended_Apnea_Start_sec": "Extended_Apnea_Start (s) (longest, rel. stim onset)",
        "Extended_Apnea_End_sec": "Extended_Apnea_End (s) (longest, rel. stim onset)",
        "N_Swallows_During_Apnea": "N_Swallows_During_Apnea (#) (all episodes)",
        "N_Extended_Apnea_Episodes": "N_Extended_Apnea_Episodes (#)",
        "Total_Apnea_Duration_sec": "Total_Apnea_Duration (s) (sum of episodes)",
        "Stim_Duration_sec": "Stim_Duration (s)",
        "Breathing_Duration_sec": "Breathing_Duration (s) (stim - total apnea)",
        "Swallow_Rate_During_Apnea_Hz": "Swallow_Rate_During_Apnea (Hz) (all episodes)",
        "Normal_IBI_PreStim_sec": "Normal_IBI_PreStim (s)",
        "Apnea_Gap_Threshold_sec": "Apnea_Gap_Threshold (s)",
        "N_Blips_Rejected_in_Stim": "N_Blips_Rejected_in_Stim (#)",
        "Resp_cycle_start_sec": "Resp_cycle_start (s)",
        "Resp_cycle_RR_peak_time_sec": "Resp_cycle_RR_peak_time (s) (T = 0)",
        "Resp_cycle_end_sec": "Resp_cycle_end (s)",
        "Swallow_RR_Lag_ms": "Swallow_RR_Lag (ms) (swallow - RR peak of its cycle)",
        "N_Swallows_Outside_Resp_Cycle": "N_Swallows_Outside_Resp_Cycle (#)",
        "SBC_Positive_ms": "SBC_Positive (ms)",
        "SBC_Negative_ms": "SBC_Negative (ms)",
        "Lag_Positive_values_ms": "Lag_Positive_values (ms)",
        "Lag_Positive_mean_ms": "Lag_Positive_mean (ms)",
        "Lag_Positive_median_ms": "Lag_Positive_median (ms)",
        "Lag_Negative_values_ms": "Lag_Negative_values (ms)",
        "Lag_Negative_mean_ms": "Lag_Negative_mean (ms)",
        "Lag_Negative_median_ms": "Lag_Negative_median (ms)",
        "N_Swallows_Lag_Measured": "N_Swallows_Lag_Measured (#)",
        "Swallow_RR_Lag_mean_ms": "Swallow_RR_Lag_mean (ms)",
        "Swallow_RR_Lag_median_ms": "Swallow_RR_Lag_median (ms)",
        "N_Lag_Positive": "N_Lag_Positive (#)",
        "Pct_Lag_Positive": "Pct_Lag_Positive (%)",
        "N_Lag_Negative": "N_Lag_Negative (#)",
        "Pct_Lag_Negative": "Pct_Lag_Negative (%)",
        "N_Lag_Zero": "N_Lag_Zero (#)",
        "Start_sec (rel. stim onset)": "Start (s) (rel. stim onset)",
        "End_sec (rel. stim onset)": "End (s) (rel. stim onset)",
        "Duration_sec": "Duration (s)",
        "N_Swallows": "N_Swallows (#)",
        "Baseline_Value_g": "Baseline_Value (g)",
        "Baseline_Peaks_Excluded": "Baseline_Peaks_Excluded (#)",
        "PreStim_Window_Start_sec (rel. stim onset)": "PreStim_Window_Start (s) (rel. stim onset)",
        "PreStim_Window_End_sec (rel. stim onset)": "PreStim_Window_End (s) (rel. stim onset)",
        "PreStim_Samples_Used_pct": "PreStim_Samples_Used (%)",
        "Excluded_Window_Duration_sec": "Excluded_Window_Duration (s)",
        "N_peaks_in_excluded_window": "N_peaks_in_excluded_window (#)",
        "Max_SG_Amplitude_WholeWindow_g": "Max_SG_Amplitude_WholeWindow (g)",
        "Avg_SG_Amplitude_WholeWindow_g": "Avg_SG_Amplitude_WholeWindow (g)",
        "Max_Tongue_Force_WholeWindow_N": "Max_Tongue_Force_WholeWindow (N)",
        "Avg_Tongue_Force_WholeWindow_N": "Avg_Tongue_Force_WholeWindow (N)",
        "First_Peak_latency_sec": "First_Peak_latency (s)",
        "N_breaths_in_excluded_window": "N_breaths_in_excluded_window (#)",
    }

    def _with_units(df):
        out = df.rename(columns=UNIT_RENAME_MAP)
        return out.rename(columns={c: _sbc_rename(c) for c in out.columns})

    excel_path = os.path.join(FIGURE_OUTPUT_DIR, "analysis_summary.xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        per_animal_readme_df().to_excel(writer, sheet_name=README_SHEET_NAME, index=False)
        _with_units(sg_summary_df).to_excel(writer, sheet_name="SG Summary (Amp & Force)", index=False)
        _with_units(baseline_correction_df).to_excel(writer, sheet_name="Baseline Correction (SG)", index=False)
        _with_units(excluded_window_df).to_excel(
            writer, sheet_name=f"Excluded Window (first {TIME_EXCLUDE:g}s)"[:31], index=False)
        _with_units(swallow_latency_df).to_excel(writer, sheet_name="Swallow Counts & Latency", index=False)
        _with_units(max_intensity_df).to_excel(writer, sheet_name="Max Intensity", index=False)
        _with_units(apnea_summary_df).to_excel(writer, sheet_name="Apnea Summary", index=False)
        _with_units(apnea_episodes_df).to_excel(writer, sheet_name="Extended Apnea Episodes", index=False)
        _with_units(sbc_summary_df).to_excel(writer, sheet_name=SBC_SHEET, index=False)
        _with_units(sbc_values_df).to_excel(writer, sheet_name=SBC_VALUES_SHEET, index=False)
        _with_units(swallow_rr_apnea_df).to_excel(writer, sheet_name=APNEA_LABEL_SWALLOW_RR, index=False)
        _with_units(stim_rr_apnea_df).to_excel(writer, sheet_name=APNEA_LABEL_STIM_RR, index=False)

        for sheet_name, sheet in writer.sheets.items():
            if sheet_name == README_SHEET_NAME:
                continue
            for col_cells in sheet.columns:
                length = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
                sheet.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 10), 40)
        format_readme_sheet(writer.sheets[README_SHEET_NAME], [26, 42, 45, 100])

    print(f"Wrote summary tables to: {os.path.abspath(excel_path)}")

    if SHOW_FIGURES:
        plt.show()
    if SAVE_FIGURES:
        print(f"Saved {_fig_counter['n']} figures to: {os.path.abspath(FIGURE_OUTPUT_DIR)}")

    # =============================================================================
    # 16. Tag key results with this animal's ID and return for pooled analysis
    # =============================================================================

    sg_summary_df_tagged = sg_summary_df.copy()
    sg_summary_df_tagged.insert(0, "Animal", animal_id)

    swallow_latency_df_tagged = swallow_latency_df.copy()
    swallow_latency_df_tagged.insert(0, "Animal", animal_id)

    max_intensity_df_tagged = max_intensity_df.copy()
    max_intensity_df_tagged.insert(0, "Animal", animal_id)

    apnea_summary_df_tagged = apnea_summary_df.copy()
    apnea_summary_df_tagged.insert(0, "Animal", animal_id)

    baseline_correction_df_tagged = baseline_correction_df.copy()
    baseline_correction_df_tagged.insert(0, "Animal", animal_id)

    excluded_window_df_tagged = excluded_window_df.copy()
    excluded_window_df_tagged.insert(0, "Animal", animal_id)

    apnea_episodes_df_tagged = apnea_episodes_df.copy()
    apnea_episodes_df_tagged.insert(0, "Animal", animal_id)

    sbc_summary_df_tagged = sbc_summary_df.copy()
    sbc_summary_df_tagged.insert(0, "Animal", animal_id)
    swallow_lags_df_tagged = sbc_values_df.copy()          # SBC values, one row per paired swallow
    swallow_lags_df_tagged.insert(0, "Animal", animal_id)

    return (sg_summary_df_tagged, swallow_latency_df_tagged,
            max_intensity_df_tagged, apnea_summary_df_tagged,
            baseline_correction_df_tagged, excluded_window_df_tagged,
            apnea_episodes_df_tagged, swallow_lags_df_tagged, sbc_summary_df_tagged)


# =============================================================================
# 16b. "READ ME" documentation sheets (what every sheet/column is and how it is
#      calculated). Written as the first sheet of every analysis_summary.xlsx and
#      of pooled_analysis_summary.xlsx. Setting values are filled in automatically.
# =============================================================================

README_SHEET_NAME = "READ ME - Definitions"
WHOLE_SHEET = "(whole sheet)"


def _topn_text():
    if NUMBER_PEAK_INCLUDE == 0:
        return "all counted peaks (NUMBER_PEAK_INCLUDE = 0)"
    return (f"the {NUMBER_PEAK_INCLUDE} highest counted peaks, or all of them if there are fewer "
            f"(NUMBER_PEAK_INCLUDE = {NUMBER_PEAK_INCLUDE})")


def _column_docs():
    """One entry per (sheet, column): sheet, raw column name, displayed name, meaning, calculation."""
    TE = f"{TIME_EXCLUDE:g}"
    CAL = f"{calibration_factor:.3f}"
    topn = _topn_text()
    sw = APNEA_LABEL_SWALLOW_RR
    st = APNEA_LABEL_STIM_RR
    D = []

    def add(sheet, raw, shown, what, how):
        D.append({"sheet": sheet, "raw": raw, "shown": shown, "what": what, "how": how})

    # ---------------- General definitions ----------------
    G = "GENERAL"
    add(G, None, "Script version", "Which version of the analysis made this file.",
        "SLN_alldata_GG_V9 = Version 8 plus the swallow-breath lag (see 'Swallow-breath lag (definition)').")
    add(G, None, "Settings used in this run", "Values of the adjustable settings when this file was made.",
        f"TIME_EXCLUDE = {TE} s; NUMBER_PEAK_INCLUDE = {NUMBER_PEAK_INCLUDE} (0 = all); calibration_factor = "
        f"{CAL} N per g; stim threshold th = {th}; pre-stim window = {tbefore_sec:g} s; "
        f"BASELINE_PEAK_PROMINENCE_FRACTION = {BASELINE_PEAK_PROMINENCE_FRACTION}; "
        f"BREATH_REFERENCE_PROMINENCE_FRACTION = {BREATH_REFERENCE_PROMINENCE_FRACTION}; "
        f"BREATH_MIN_PROMINENCE_FRACTION = {BREATH_MIN_PROMINENCE_FRACTION}; "
        f"BLIP_MIN_PROMINENCE_FRACTION = {BLIP_MIN_PROMINENCE_FRACTION}; "
        f"BREATH_MIN_SPACING_FRACTION = {BREATH_MIN_SPACING_FRACTION}; "
        f"VERIFIED_APNEA_MULTIPLIER = {VERIFIED_APNEA_MULTIPLIER}; "
        f"BREATH_CYCLE_BASELINE_FRACTION = {BREATH_CYCLE_BASELINE_FRACTION}; "
        f"EXTENDED_APNEA_IBI_MULTIPLIER = {EXTENDED_APNEA_IBI_MULTIPLIER}; "
        f"EXTENDED_APNEA_MIN_SEC = {EXTENDED_APNEA_MIN_SEC} s.")
    add(G, None, "Condition", "Which recording the row belongs to.",
        "A30, A40, D30, D40 = the stimulated target conditions (number = stimulation intensity in uA; "
        "A = the A10->A40 series, D = the D40->D10 series). BS1, BS2, BS3 = baseline recordings. "
        "Conditions are found from the LabChart comment text.")
    add(G, None, "Block", "Which repeat of the condition.",
        "Block 1 = the first time the condition comment appears, Block 2 = the second time. On plots, "
        "Block 1 is drawn blue and Block 2 red.")
    add(G, None, "Stim window / stim onset / stim offset",
        "The period when stimulation was actually delivered.",
        f"Each stimulated condition is cut from {tbefore_sec:g} s before the stimulus train to after it. "
        f"Stim onset = first sample of the STIM channel above th ({th}); stim offset = last sample above th. "
        "Every time marked 'rel. stim onset' uses onset = 0 s.")
    add(G, None, "Swallow (SG peak) detection",
        "How swallows are found on the strain-gauge (SG) signal.",
        f"On the baseline-corrected SG segment, peaks are found with scipy find_peaks. Minimum prominence = "
        f"Otsu threshold of all peak prominences in the segment x {PROMINENCE_SAFETY_MARGIN}; minimum spacing = "
        "0.35 x the median spacing of those peaks. A 'ringing' filter then removes the smaller of two "
        "neighbouring peaks if the signal does not fall at least halfway back to the segment's 10th-percentile "
        "level between them. Only peaks between stim onset and stim offset are kept.")
    add(G, None, "Counted peaks vs. excluded window",
        "Which swallow peaks enter the main results.",
        f"Peaks in the first {TE} s after stim onset (TIME_EXCLUDE) are in the EXCLUDED WINDOW and are not "
        f"counted in the main SG results; they are reported separately on the 'Excluded Window' sheet. "
        f"COUNTED peaks = peaks from onset + {TE} s to stim offset.")
    add(G, None, "Amplitude (g) and tongue force (N)", "Units of SG results.",
        f"SG amplitudes are in g, measured from the pre-stim baseline (baseline-corrected). "
        f"Tongue force (N) = amplitude (g) x calibration_factor ({CAL} = 1.66 / 5).")
    add(G, None, "SG baseline correction", "How the SG baseline is removed.",
        f"For every target condition and block: take the {tbefore_sec:g} s before stim onset; find peaks there "
        f"(prominence >= {BASELINE_PEAK_PROMINENCE_FRACTION} x the swallow-detection threshold) and cut each "
        f"one out over its width at {BASELINE_PEAK_WIDTH_REL_HEIGHT:g} of its prominence; baseline = median of "
        "the remaining samples. That value is subtracted from the whole segment of that block. Always applied.")
    add(G, None, "Breath detection (RR)", "How breaths and blips are found on the respiratory (RR) signal.",
        f"Normal breathing reference = the {tbefore_sec:g} s before stim onset: peaks with prominence >= "
        f"{BREATH_REFERENCE_PROMINENCE_FRACTION} x "
        "that window's 5th-95th percentile range; their median prominence = normal breath size, their median "
        "spacing = normal inter-breath interval (IBI). Then all RR peaks with prominence >= "
        f"{BLIP_MIN_PROMINENCE_FRACTION} x normal breath size and at least {BREATH_MIN_SPACING_FRACTION} x normal "
        f"IBI apart are found. Peaks >= {BREATH_MIN_PROMINENCE_FRACTION} x normal breath size = REAL BREATHS "
        "(blue dots). Smaller peaks = BLIPS: not breaths, never counted, shown as grey crosses. If there is no "
        "usable pre-stim breathing, the whole segment is used as the reference.")
    add(G, None, "Extended apnea (definition)", "A stretch of the stim window without normal breathing.",
        f"Gap threshold = max({EXTENDED_APNEA_MIN_SEC:g} s, {EXTENDED_APNEA_IBI_MULTIPLIER:g} x normal pre-stim "
        "IBI). A real breath is 'normal breathing' if another real breath is within the gap threshold of it; an "
        "isolated single breath or a blip does NOT end an apnea. An extended apnea episode = time between two "
        "normal-breathing breaths that is >= the gap threshold, clipped to the stim window (so an apnea already "
        "present at onset starts at 0 s, and one still going at offset ends at stim offset). An animal can have "
        "several episodes. Boundaries are the dark-green lines on the RR plots.")
    add(G, None, "Swallow-breath lag (definition)",
        "Timing of each swallow relative to the nearest breath, in MILLISECONDS (ms).",
        "Only swallows inside a RESPIRATORY CYCLE are used. Each real breath's cycle runs from where the RR trace "
        "rises off baseline to where it returns to baseline (within BREATH_CYCLE_BASELINE_FRACTION = "
        f"{BREATH_CYCLE_BASELINE_FRACTION} of the breath's height above its base); the breath's RR peak is T = 0. "
        "Lag (ms) = (swallow time - RR peak time of that cycle) x 1000. Swallow time = the strain-gauge swallow "
        "peak (red dashed line on the RR plots); blips are never breaths. Positive = swallow after the RR peak "
        "(expiration); negative = before the RR peak (inspiration). Swallows outside every cycle (flat baseline "
        "between breaths, apnea) get no lag. ONE SWALLOW PER BREATH: if several swallows fall in the same cycle, "
        "only the one closest to the RR peak gets a lag; the others are Paired_with_breath = False. "
        "ON THE RR PLOTS each paired swallow has a star at the top of its red dashed line: BLUE = positive lag, "
        "PINK = negative lag, DARK GREEN = zero lag (same time as the RR peak); the legend gives the count of each. "
        "Considered: ALL detected swallows in the stim window, "
        f"including the first {TE} s and swallows during extended apnea; these are flagged in the "
        "In_excluded_window and During_extended_apnea columns. No distance limit is applied. Per-swallow values: "
        f"'{sw}' sheet and '{SBC_VALUES_SHEET}' sheet; per-block summary: '{SBC_SHEET}' sheet.")

    # ---------------- SG Summary ----------------
    S = "SG Summary (Amp & Force)"
    add(S, None, WHOLE_SHEET, "Strain-gauge amplitude and tongue force per condition and block.",
        "Baseline rows (BS1-3) are quality-control only. Stimulated rows use baseline-corrected SG.")
    add(S, "Role", "Role", "Type of row.", "'Baseline (QC only)' = BS1-3 recordings; 'Stimulated' = target conditions.")
    add(S, "Baseline_Value_g", "Baseline_Value (g)", "SG baseline subtracted from this block (raw g).",
        "Median of the peak-free pre-stim samples (see GENERAL > SG baseline correction and the "
        "'Baseline Correction (SG)' sheet). Blank for BS rows.")
    add(S, "Baseline_Peaks_Excluded", "Baseline_Peaks_Excluded (#)",
        "Number of peaks left out of the pre-stim baseline.", "Count of peaks overlapping the pre-stim window.")
    add(S, "Baseline_Correction_Applied", "Baseline_Correction_Applied",
        "Whether the baseline was subtracted.",
        "True for every stimulated block with a pre-stim window. False for BS rows (not corrected).")
    add(S, "Max_SG_Amplitude_g", "Max_SG_Amplitude (g)", "Largest swallow amplitude.",
        f"Stimulated rows: highest peak among {topn}. BS rows: highest raw SG value over the whole baseline "
        "recording (START to END comment), not baseline-corrected.")
    add(S, "Avg_SG_Amplitude_g", "Avg_SG_Amplitude (g)", "Average swallow amplitude.",
        f"Stimulated rows: mean of {topn}. BS rows: mean of the raw SG signal over the whole baseline recording.")
    add(S, "Max_Tongue_Force_N", "Max_Tongue_Force (N)", "Largest tongue force.", f"Max_SG_Amplitude x {CAL}.")
    add(S, "Avg_Tongue_Force_N", "Avg_Tongue_Force (N)", "Average tongue force.", f"Avg_SG_Amplitude x {CAL}.")
    add(S, "Max_SG_Amplitude_WholeTrain_g", "Max_SG_Amplitude_WholeTrain (g)",
        "Highest SG value during the whole stimulus train.",
        "Maximum of every baseline-corrected SG sample from stim onset to stim offset (all samples, not only "
        f"peaks; includes the first {TE} s).")
    add(S, "Avg_SG_Amplitude_WholeTrain_g", "Avg_SG_Amplitude_WholeTrain (g)",
        "Mean SG level during the whole stimulus train.",
        "Mean of every baseline-corrected SG sample from stim onset to stim offset.")
    add(S, "Max_Tongue_Force_WholeTrain_N", "Max_Tongue_Force_WholeTrain (N)", "Whole-train max as force.",
        f"Max_SG_Amplitude_WholeTrain x {CAL}.")
    add(S, "Avg_Tongue_Force_WholeTrain_N", "Avg_Tongue_Force_WholeTrain (N)", "Whole-train mean as force.",
        f"Avg_SG_Amplitude_WholeTrain x {CAL}.")

    # ---------------- Baseline Correction ----------------
    S = "Baseline Correction (SG)"
    add(S, None, WHOLE_SHEET, "Details of the SG baseline for every target condition and block.",
        "Use together with the dotted baseline line and grey crosses on the combined SG + RR summary plots.")
    add(S, "PreStim_Window_Start_sec (rel. stim onset)", "PreStim_Window_Start (s) (rel. stim onset)",
        "Start of the window used for the baseline.", f"Normally -{tbefore_sec:g} s.")
    add(S, "PreStim_Window_End_sec (rel. stim onset)", "PreStim_Window_End (s) (rel. stim onset)",
        "End of the window used for the baseline.", "Stim onset (0 s).")
    add(S, "Baseline_Peaks_Excluded", "Baseline_Peaks_Excluded (#)", "Peaks removed from the window.",
        f"Peaks with prominence >= {BASELINE_PEAK_PROMINENCE_FRACTION} x the swallow-detection threshold that "
        "overlap the window.")
    add(S, "PreStim_Samples_Used_pct", "PreStim_Samples_Used (%)", "How much of the window was used.",
        "Samples left after cutting out the peaks / all samples in the window x 100.")
    add(S, "Baseline_Value_g", "Baseline_Value (g)", "The baseline value (raw g).",
        "Median of the samples left after removing peaks (median of the whole window if nothing was left).")
    add(S, "Correction_Applied", "Correction_Applied", "Whether it was subtracted.",
        "True whenever a baseline could be computed.")

    # ---------------- Excluded Window ----------------
    S = f"Excluded Window (first {TE}s)"
    add(S, None, WHOLE_SHEET, f"Results for the first {TE} s after stim onset (the TIME_EXCLUDE window).",
        f"Window = stim onset to onset + {TE} s (or stim offset if earlier). Same calculations as the counted "
        "part of the block, using only peaks and signal inside this window.")
    add(S, "Excluded_Window_Duration_sec", "Excluded_Window_Duration (s)", "Length of the window.",
        f"min({TE} s, stim duration).")
    add(S, "N_peaks_in_excluded_window", "N_peaks_in_excluded_window (#)", "Swallow peaks in the window.",
        "Number of detected swallow peaks between onset and onset + TIME_EXCLUDE.")
    add(S, "Max_SG_Amplitude_g", "Max_SG_Amplitude (g)", "Largest peak in the window.",
        f"Highest peak among the window's peaks (same top-N rule: {topn.replace('counted ', '')}).")
    add(S, "Avg_SG_Amplitude_g", "Avg_SG_Amplitude (g)", "Average peak height in the window.",
        "Mean of the same selected peaks.")
    add(S, "Max_Tongue_Force_N", "Max_Tongue_Force (N)", "Largest peak as force.", f"Max_SG_Amplitude x {CAL}.")
    add(S, "Avg_Tongue_Force_N", "Avg_Tongue_Force (N)", "Average peak as force.", f"Avg_SG_Amplitude x {CAL}.")
    add(S, "Max_SG_Amplitude_WholeWindow_g", "Max_SG_Amplitude_WholeWindow (g)", "Highest SG sample in the window.",
        "Maximum of every baseline-corrected SG sample in the window.")
    add(S, "Avg_SG_Amplitude_WholeWindow_g", "Avg_SG_Amplitude_WholeWindow (g)", "Mean SG level in the window.",
        "Mean of every baseline-corrected SG sample in the window.")
    add(S, "Max_Tongue_Force_WholeWindow_N", "Max_Tongue_Force_WholeWindow (N)", "As force.",
        f"Max_SG_Amplitude_WholeWindow x {CAL}.")
    add(S, "Avg_Tongue_Force_WholeWindow_N", "Avg_Tongue_Force_WholeWindow (N)", "As force.",
        f"Avg_SG_Amplitude_WholeWindow x {CAL}.")
    add(S, "First_Peak_latency_sec", "First_Peak_latency (s)", "Time to the first peak.",
        "Time of the first peak in the window - stim onset.")
    add(S, "Swallow_Rate_Hz", "Swallow_Rate (Hz)", "Swallows per second in the window.",
        "N_peaks_in_excluded_window / Excluded_Window_Duration.")
    add(S, "Inter_Swallow_Interval_ms", "Inter_Swallow_Interval (ms)", "Average time between peaks.",
        "Mean interval between consecutive peaks in the window x 1000 (needs >= 2 peaks).")
    add(S, "N_breaths_in_excluded_window", "N_breaths_in_excluded_window (#)", "Real breaths in the window.",
        "Number of real breaths (blips not counted) between onset and onset + TIME_EXCLUDE.")
    add(S, f"{sw}_rate_assumed", f"{sw}_rate_assumed (#)", "Swallow-RR apneas in the window (count).",
        f"Number of swallows in the window that have a '{sw}' row.")
    add(S, f"{sw}_rate_verified", f"{sw}_rate_verified (#)", "Verified ones among them.",
        f"Count of those with Verified_apnea = True on the '{sw}' sheet.")
    add(S, f"{sw}_first_latency_sec", f"{sw}_first_latency (s)", "Latency of the first one.",
        "Apnea_latency of the first swallow in the window.")
    add(S, f"{sw}_mean_duration_sec", f"{sw}_mean_duration (s)", "Mean apnea duration after these swallows.",
        "Mean of Apnea_duration (swallow to next real breath) for swallows in the window.")

    # ---------------- Swallow Counts & Latency ----------------
    S = "Swallow Counts & Latency"
    add(S, None, WHOLE_SHEET, "Swallow counts, latency, rate and interval per condition and block.", "")
    add(S, "N_swallows_in_stim_window", "N_swallows_in_stim_window (#)", "Counted swallows.",
        f"Number of counted peaks (from onset + {TE} s to stim offset).")
    add(S, "N_swallows_full_20sec", "N_swallows_full_20sec (#)", "All swallows during the stimulus.",
        f"Number of detected peaks from stim onset to stim offset, including the first {TE} s.")
    add(S, "Max_Peak_In_Excluded_Window_g", "Max_Peak_In_Excluded_Window (g)", "Largest peak in the excluded window.",
        f"Highest peak amplitude in the first {TE} s (blank if none).")
    add(S, "Swallow_latency_sec", "Swallow_latency (s)", "Time to the first swallow.",
        f"Time of the first detected peak (including the first {TE} s) - stim onset.")
    add(S, "Messy_Signal_Duration_sec", "Messy_Signal_Duration (s)", "Swallow latency, capped.",
        f"min(Swallow_latency, {MESSY_SIGNAL_WINDOW_SEC:g} s); {MESSY_SIGNAL_WINDOW_SEC:g} s if there was no swallow.")
    add(S, "Messy_Signal_Duration_Uncapped_ms", "Messy_Signal_Duration_Uncapped (ms)", "Swallow latency, not capped.",
        "Swallow_latency x 1000.")
    add(S, "Swallow_Rate_Hz", "Swallow_Rate (Hz)", "Swallows per second (counted part).",
        f"N_swallows_in_stim_window / (stim offset - onset - {TE} s).")
    add(S, "Inter_Swallow_Interval_ms", "Inter_Swallow_Interval (ms)", "Average time between counted swallows.",
        "Mean interval between consecutive counted peaks x 1000 (needs >= 2 peaks).")

    # ---------------- Max Intensity ----------------
    S = "Max Intensity"
    add(S, None, WHOLE_SHEET, "Which target condition gave the strongest tongue force, per block.", "")
    add(S, "Condition_at_max_force", "Condition_at_max_force", "Winning condition.",
        "Target condition (A30, A40, D30, D40) with the highest Max_Tongue_Force (top-N peaks) in that block.")
    add(S, "Intensity_uA", "Intensity (uA)", "Its stimulation intensity.", "Number in the condition name (e.g. A40 -> 40).")
    add(S, "Max_Tongue_Force_N", "Max_Tongue_Force (N)", "That force.", "Max_Tongue_Force of the winning condition.")

    # ---------------- Apnea Summary ----------------
    S = "Apnea Summary"
    add(S, None, WHOLE_SHEET, "Respiratory (apnea) results per condition and block.",
        "Uses real breaths only (blips are ignored). See GENERAL for breath and extended-apnea definitions.")
    add(S, f"{sw}_rate_assumed", f"{sw}_rate_assumed (#)", "Number of swallow-related apneas (a count).",
        f"Number of rows on the '{sw}' sheet for this block = every detected swallow in the stim window "
        "(each swallow is assumed to cause an apnea).")
    add(S, f"{sw}_rate_verified", f"{sw}_rate_verified (#)", "Number of verified swallow-related apneas.",
        f"Count of those rows with Verified_apnea = True.")
    add(S, f"{sw}_first_latency_sec", f"{sw}_first_latency (s)", "Latency of the first swallow-related apnea.",
        "Apnea_latency of the first swallow (first swallow time - stim onset).")
    add(S, f"{sw}_mean_duration_sec", f"{sw}_mean_duration (s)", "Mean swallow-related apnea duration.",
        "Mean of Apnea_duration (swallow to next real breath) over all swallows in the block.")
    add(S, f"{st}_duration_sec", f"{st}_duration (s)", "Apnea caused by stim onset.",
        f"Copied from the '{st}' sheet: first real breath after onset - onset.")
    add(S, f"{st}_verified", f"{st}_verified", "Whether that apnea is verified.", f"Copied from the '{st}' sheet.")
    add(S, "Extended_Apnea_Duration_sec", "Extended_Apnea_Duration (s) (longest episode)",
        "Length of the longest extended apnea episode.", "End - start of the longest episode (blank if none).")
    add(S, "Extended_Apnea_Start_sec", "Extended_Apnea_Start (s) (longest, rel. stim onset)",
        "Start of the longest episode.", "Relative to stim onset.")
    add(S, "Extended_Apnea_End_sec", "Extended_Apnea_End (s) (longest, rel. stim onset)",
        "End of the longest episode.", "Relative to stim onset.")
    add(S, "N_Swallows_During_Apnea", "N_Swallows_During_Apnea (#) (all episodes)",
        "Swallows that happened during extended apnea.",
        f"Number of detected swallows (including the first {TE} s) inside any extended apnea episode.")
    add(S, "N_Extended_Apnea_Episodes", "N_Extended_Apnea_Episodes (#)", "How many extended apneas.",
        "Number of episodes in the stim window (see 'Extended Apnea Episodes' sheet).")
    add(S, "Total_Apnea_Duration_sec", "Total_Apnea_Duration (s) (sum of episodes)", "Total time in extended apnea.",
        "Sum of the durations of all episodes.")
    add(S, "Stim_Duration_sec", "Stim_Duration (s)", "Length of the stimulus train.",
        "Stim offset - stim onset (about 20 s).")
    add(S, "Breathing_Duration_sec", "Breathing_Duration (s) (stim - total apnea)",
        "Time during the stimulus when the animal was breathing.", "Stim_Duration - Total_Apnea_Duration.")
    add(S, "Swallow_Rate_During_Apnea_Hz", "Swallow_Rate_During_Apnea (Hz) (all episodes)",
        "Swallows per second while in extended apnea.", "N_Swallows_During_Apnea / Total_Apnea_Duration.")
    add(S, "Normal_IBI_PreStim_sec", "Normal_IBI_PreStim (s)", "Normal time between breaths.",
        f"Median interval between breaths in the {tbefore_sec:g} s before stim onset.")
    add(S, "Apnea_Gap_Threshold_sec", "Apnea_Gap_Threshold (s)", "Minimum length of an extended apnea.",
        f"max({EXTENDED_APNEA_MIN_SEC:g} s, {EXTENDED_APNEA_IBI_MULTIPLIER:g} x Normal_IBI_PreStim).")
    add(S, "N_Blips_Rejected_in_Stim", "N_Blips_Rejected_in_Stim (#)", "Tiny RR bumps not counted as breaths.",
        "Number of blips (grey crosses) between stim onset and offset.")
    lag_note = (f"Uses the paired swallows (one per breath) among ALL detected swallows in the stim window "
                f"(including the first {TE} s and swallows during extended apnea); per-swallow values and flags "
                f"are on the '{sw}' sheet.")
    add(S, "N_Swallows_Lag_Measured", "N_Swallows_Lag_Measured (#)", "Swallows with a swallow-breath lag.",
        "Number of swallows paired with a breath (Paired_with_breath = True) = number of breaths with a swallow "
        "lag. " + lag_note)
    add(S, "Swallow_RR_Lag_mean_ms", "Swallow_RR_Lag_mean (ms)", "Average swallow-breath lag, in milliseconds.",
        "Mean of Swallow_RR_Lag (ms) over the block's swallows. " + lag_note)
    add(S, "Swallow_RR_Lag_median_ms", "Swallow_RR_Lag_median (ms)", "Median swallow-breath lag, in milliseconds.",
        "Median of Swallow_RR_Lag (ms) over the block's swallows.")
    add(S, "N_Lag_Positive", "N_Lag_Positive (#)", "Swallows after the nearest RR peak (expiration).",
        "Number of swallows with Swallow_RR_Lag > 0 ms.")
    add(S, "Pct_Lag_Positive", "Pct_Lag_Positive (%)", "Share of swallows with a positive lag.",
        "N_Lag_Positive / N_Swallows_Lag_Measured x 100.")
    add(S, "N_Lag_Negative", "N_Lag_Negative (#)", "Swallows before the nearest RR peak (inspiration).",
        "Number of swallows with Swallow_RR_Lag < 0 ms.")
    add(S, "Pct_Lag_Negative", "Pct_Lag_Negative (%)", "Share of swallows with a negative lag.",
        "N_Lag_Negative / N_Swallows_Lag_Measured x 100.")
    add(S, "N_Lag_Zero", "N_Lag_Zero (#)", "Swallows exactly on an RR peak.",
        "Number of swallows with Swallow_RR_Lag = 0 ms (counted in neither positive nor negative).")
    add(S, "Lag_Positive_values_ms", "Lag_Positive_values (ms)", "Every positive lag in the block (blue stars).",
        "The Swallow_RR_Lag values > 0 ms, in time order, separated by commas (1 decimal).")
    add(S, "Lag_Positive_mean_ms", "Lag_Positive_mean (ms)", "Average of the positive lags.",
        "Mean of Lag_Positive_values (blank if none).")
    add(S, "Lag_Positive_median_ms", "Lag_Positive_median (ms)", "Median of the positive lags.",
        "Median of Lag_Positive_values (blank if none).")
    add(S, "Lag_Negative_values_ms", "Lag_Negative_values (ms)", "Every negative lag in the block (pink stars).",
        "The Swallow_RR_Lag values < 0 ms, in time order, separated by commas (1 decimal).")
    add(S, "Lag_Negative_mean_ms", "Lag_Negative_mean (ms)", "Average of the negative lags.",
        "Mean of Lag_Negative_values (blank if none); a negative number.")
    add(S, "Lag_Negative_median_ms", "Lag_Negative_median (ms)", "Median of the negative lags.",
        "Median of Lag_Negative_values (blank if none).")
    add(S, "N_Swallows_Outside_Resp_Cycle", "N_Swallows_Outside_Resp_Cycle (#)",
        "Swallows not inside any breath's respiratory cycle.",
        "Detected swallows on flat baseline between breaths or in apnea (In_respiratory_cycle = False); no lag. "
        "Swallows during each extended apnea are counted per episode on the 'Extended Apnea Episodes' sheet.")

    # ---------------- Extended Apnea Episodes ----------------
    S = "Extended Apnea Episodes"
    add(S, None, WHOLE_SHEET, "One row per extended apnea episode.",
        "Blocks with no episode have no rows (their N_Extended_Apnea_Episodes is 0 on 'Apnea Summary').")
    add(S, "Episode", "Episode", "Episode number within the block.", "1, 2, ... in time order.")
    add(S, "Start_sec (rel. stim onset)", "Start (s) (rel. stim onset)", "Start of the episode.",
        "Last normal-breathing breath before the pause, or stim onset if the pause began before onset.")
    add(S, "End_sec (rel. stim onset)", "End (s) (rel. stim onset)", "End of the episode.",
        "First breath of resumed normal breathing, or stim offset.")
    add(S, "Duration_sec", "Duration (s)", "Length of the episode.", "End - Start.")
    add(S, "Ended_by", "Ended_by", "Why the episode ended.",
        "'Breathing resumed' or 'Stim offset' (apnea still going when stimulation stopped).")
    add(S, "N_Swallows", "N_Swallows (#)", "Swallows during the episode.",
        f"Detected swallows (including the first {TE} s) between Start and End.")
    add(S, "Swallow_Rate_Hz", "Swallow_Rate (Hz)", "Swallow rate during the episode.", "N_Swallows / Duration.")

    # ---------------- Swallow-RR Apnea ----------------
    S = sw
    add(S, None, WHOLE_SHEET, "One row per detected swallow: the breathing pause that follows it.",
        f"Includes every detected swallow in the stim window (also the first {TE} s).")
    add(S, "Swallow_number", "Swallow_number", "Swallow order in the block.", "1, 2, ... in time order.")
    add(S, "Swallow_time_sec", "Swallow_time (s)", "When the swallow happened.",
        f"Seconds from the START OF THE SEGMENT (which begins about {tbefore_sec:g} s before stim onset), "
        "not from stim onset.")
    add(S, "Apnea_latency_sec (time from stim onset to swallow)", "Apnea_latency (s) (time from stim onset to swallow)",
        "Time from stim onset to the swallow.", "Swallow time - stim onset.")
    add(S, "Apnea_duration_sec (assumed, to next breath)", "Apnea_duration (s) (assumed, to next breath)",
        "Assumed apnea after the swallow.", "First real breath after the swallow - swallow time.")
    add(S, "Local_normal_IBI_sec", "Local_normal_IBI (s)", "Local normal breath interval.",
        "Median interval between real breaths within +/-10 s of the swallow (all breaths if fewer than 3 there).")
    add(S, "Verified_apnea", "Verified_apnea", "Whether the pause is a real apnea.",
        f"True if Apnea_duration > {VERIFIED_APNEA_MULTIPLIER:g} x Local_normal_IBI.")
    add(S, "In_respiratory_cycle", "In_respiratory_cycle", "Whether the swallow falls inside a breath's cycle.",
        "True if the swallow is between the start and end of a real breath's respiratory cycle; False if it is on "
        "flat baseline between breaths or in apnea (then no lag).")
    add(S, "Resp_cycle_start_sec", "Resp_cycle_start (s)", "Start of the swallow's respiratory cycle.",
        f"Where the RR trace rises off baseline: the breath is within {BREATH_CYCLE_BASELINE_FRACTION} of its height "
        "above its base. Seconds from the start of the segment (same reference as Swallow_time).")
    add(S, "Resp_cycle_RR_peak_time_sec", "Resp_cycle_RR_peak_time (s) (T = 0)", "RR peak of that cycle (T = 0).",
        "Time of the real breath peak (blue dot) of the cycle the swallow falls in.")
    add(S, "Resp_cycle_end_sec", "Resp_cycle_end (s)", "End of the swallow's respiratory cycle.",
        "Where the RR trace returns to baseline (same rule as the start).")
    add(S, "Swallow_RR_Lag_ms", "Swallow_RR_Lag (ms) (swallow - RR peak of its cycle)",
        "Swallow-breath lag, in MILLISECONDS.",
        "(Swallow_time - Resp_cycle_RR_peak_time) x 1000. Positive = swallow after the RR peak (expiration); "
        "negative = before the RR peak (inspiration). Blank if the swallow is outside every cycle, or if another "
        "swallow in the same cycle is closer to the RR peak (see Paired_with_breath).")
    add(S, "Paired_with_breath", "Paired_with_breath", "Whether this swallow got a lag.",
        "True if the swallow is inside a respiratory cycle and is the closest swallow to that cycle's RR peak. "
        "False if it is outside every cycle, or another swallow in the same cycle is closer to the RR peak (one "
        "swallow per breath); then Swallow_RR_Lag and Lag_sign are blank.")
    add(S, "Lag_sign", "Lag_sign", "Sign of the lag.",
        "'Positive' (> 0 ms), 'Negative' (< 0 ms) or 'Zero'. Blank if Paired_with_breath = False.")
    add(S, "Plot_star_color", "Plot_star_color", "Colour of this swallow's star on the RR plots.",
        "Blue = positive lag (swallow after the RR peak); Pink = negative lag (before the RR peak); Dark green = "
        "zero lag (same time as the RR peak). Blank = no lag, so no star is drawn.")
    add(S, "In_excluded_window", "In_excluded_window", "Whether the swallow is in the first TIME_EXCLUDE seconds.",
        f"True if the swallow is within the first {TE} s after stim onset.")
    add(S, "During_extended_apnea", "During_extended_apnea", "Whether the swallow happened during an extended apnea.",
        "True if the swallow falls inside an extended apnea episode (see 'Extended Apnea Episodes' sheet).")

    # ---------------- Stim-RR Apnea ----------------
    S = st
    add(S, None, WHOLE_SHEET, "One row per block: the breathing pause starting at stim onset.", "")
    add(S, "Apnea_latency_sec (time from stim onset; = 0 by definition)",
        "Apnea_latency (s) (time from stim onset; = 0 by definition)", "Latency.", "Always 0 (measured from onset).")
    add(S, "Apnea_duration_sec (assumed, stim onset to next breath)",
        "Apnea_duration (s) (assumed, stim onset to next breath)", "Pause after stim onset.",
        "First real breath after stim onset - stim onset.")
    add(S, "Local_normal_IBI_sec", "Local_normal_IBI (s)", "Local normal breath interval.",
        "Median interval between real breaths within +/-10 s of stim onset.")
    add(S, "Verified_apnea", "Verified_apnea", "Whether the pause is a real apnea.",
        f"True if Apnea_duration > {VERIFIED_APNEA_MULTIPLIER:g} x Local_normal_IBI.")
    # Swallow-Breathing Coordination (SBC): summary columns live on their own sheet
    for d in D:
        if d["sheet"] == "Apnea Summary" and d["raw"] in SBC_SUMMARY_COLS:
            d["sheet"] = SBC_SHEET
    D.append({"sheet": SBC_SHEET, "raw": None, "shown": WHOLE_SHEET,
              "what": "Swallow-Breathing Coordination (SBC) summary per condition and block.",
              "how": "SBC = swallow time (SG peak) - RR peak (T = 0) of the respiratory cycle the swallow falls in, in "
                     "milliseconds. Positive values stay positive (swallow after the RR peak) and negative values stay "
                     "negative (swallow before the RR peak). Uses one swallow per breath; see GENERAL for the full "
                     f"definition and the '{SBC_VALUES_SHEET}' sheet for every individual value."})
    V = SBC_VALUES_SHEET
    for raw, shown, what, how in [
        (None, WHOLE_SHEET, "Every individual SBC value, one row per paired swallow (one per star on the RR plots).",
         "Positive and negative values are in separate columns and keep their sign. Swallows without an SBC value "
         "(outside every respiratory cycle, or not the closest in their cycle) are not listed."),
        ("Swallow_number", "Swallow_number", "Swallow order in the block.", "Same numbering as the 'Swallow-RR Apnea' sheet."),
        ("Swallow_time_sec", "Swallow_time (s)", "When the swallow happened.", "Seconds from the start of the segment."),
        ("Resp_cycle_RR_peak_time_sec", "Resp_cycle_RR_peak_time (s) (T = 0)", "RR peak of the swallow's cycle.",
         "Same time reference as Swallow_time."),
        ("SBC_Positive_ms", "SBC_Positive (ms)", "SBC value if POSITIVE (swallow after the RR peak; blue star).",
         "(Swallow_time - Resp_cycle_RR_peak_time) x 1000 when > 0; blank otherwise. Kept positive."),
        ("SBC_Negative_ms", "SBC_Negative (ms)", "SBC value if NEGATIVE (swallow before the RR peak; pink star).",
         "(Swallow_time - Resp_cycle_RR_peak_time) x 1000 when < 0; blank otherwise. Kept negative."),
        ("SBC_Zero", "SBC_Zero", "Whether the SBC value is exactly 0 ms (dark green star).", "True if swallow time = RR peak time."),
        ("Plot_star_color", "Plot_star_color", "Colour of the star on the RR plots.", "Blue, Pink or Dark green."),
        ("In_excluded_window", "In_excluded_window", "Whether the swallow is in the first TIME_EXCLUDE seconds.",
         "True if within the excluded window after stim onset."),
        ("During_extended_apnea", "During_extended_apnea", "Whether the swallow happened during an extended apnea.",
         "True if inside an extended apnea episode."),
    ]:
        D.append({"sheet": V, "raw": raw, "shown": shown, "what": what, "how": how})
    for d in D:
        for key in ("shown", "what", "how"):
            d[key] = _sbc_rename(d[key])
    return D


def per_animal_readme_df():
    rows = []
    for d in _column_docs():
        rows.append({"Sheet": d["sheet"], "Column": d["shown"],
                     "What it is": d["what"], "How it is calculated": d["how"]})
    return pd.DataFrame(rows, columns=["Sheet", "Column", "What it is", "How it is calculated"])


def pooled_readme_df(pivot_groups, stacked_sheets):
    """pivot_groups: list of (param_list, {id(source_df): per-animal sheet name}).
    stacked_sheets: list of (pooled sheet name, per-animal sheet name, description)."""
    docs = {(d["sheet"], d["raw"]): d for d in _column_docs() if d["raw"] is not None}
    rows = [
        {"Pooled sheet": "(all sheets)", "Contents": "What the pooled file is.",
         "Source (per-animal sheet -> column)": "Every animal's figures_output/analysis_summary.xlsx",
         "How it is calculated": "Results of every successfully processed animal combined into one file. Animals "
         "that were skipped (missing files or an error) are not included - the terminal output lists them."},
        {"Pooled sheet": "(sheets ending _B1 / _B2)", "Contents": "Layout of the per-parameter sheets.",
         "Source (per-animal sheet -> column)": "",
         "How it is calculated": "Rows = animal ID; columns = A40, D40, A30, D30; each cell = that animal's value "
         "for that condition from the listed per-animal column. _B1 = Block 1, _B2 = Block 2. Nothing is averaged: "
         "each cell is the single per-animal value."},
        {"Pooled sheet": "(sheets starting SBC_)", "Contents": "Swallow-Breathing Coordination (SBC), Version 9.",
         "Source (per-animal sheet -> column)": "Apnea Summary",
         "How it is calculated": "Swallow time (SG peak) minus the RR peak (T = 0) of the respiratory cycle the "
         "swallow falls in, in milliseconds (ms); positive = after the RR peak, negative = before it. Swallows "
         "outside every cycle are not used; one swallow per breath (the closest to the RR peak). "
         "Each sheet is listed below with its exact source column."},
    ]
    note_top10 = (f" (sheet name says 'Top10' but the number of peaks used is NUMBER_PEAK_INCLUDE = "
                  f"{NUMBER_PEAK_INCLUDE})" if NUMBER_PEAK_INCLUDE != 10 else "")
    for param_list, sheet_of in pivot_groups:
        for name_prefix, source_df, value_col, unit in param_list:
            src_sheet = sheet_of.get(id(source_df), "?")
            lookup = src_sheet if not src_sheet.startswith("Excluded Window") else f"Excluded Window (first {TIME_EXCLUDE:g}s)"
            d = docs.get((lookup, value_col))
            unit_tag = f" ({unit})" if unit else ""
            shown = d["shown"] if d else value_col
            rows.append({
                "Pooled sheet": f"{name_prefix}_B1{unit_tag} / {name_prefix}_B2{unit_tag}"
                                + (note_top10 if "Top10" in name_prefix else ""),
                "Contents": d["what"] if d else "",
                "Source (per-animal sheet -> column)": f"{src_sheet} -> {shown}",
                "How it is calculated": d["how"] if d else "",
            })
    for pooled_name, src_sheet, desc in stacked_sheets:
        rows.append({"Pooled sheet": pooled_name, "Contents": desc,
                     "Source (per-animal sheet -> column)": f"{src_sheet} (all columns)",
                     "How it is calculated": f"Every animal's '{src_sheet}' sheet stacked, with an 'Animal' column "
                     "added. Column meanings are the same as in the per-animal file (see its READ ME sheet)."})
    rows = [{k: _sbc_rename(v) for k, v in r.items()} for r in rows]
    return pd.DataFrame(rows, columns=["Pooled sheet", "Contents",
                                       "Source (per-animal sheet -> column)", "How it is calculated"])


def format_readme_sheet(ws, widths):
    """Wrap text, set widths, bold the header, freeze it, and highlight section rows."""
    from openpyxl.styles import Alignment, Font, PatternFill
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w
    fill = PatternFill("solid", start_color="DDEBF7")
    for row in ws.iter_rows():
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        if row[1].value == WHOLE_SHEET or row[0].value in ("(all sheets)", "(sheets ending _B1 / _B2)", "(sheets starting SBC_)"):
            for c in row:
                c.fill = fill
                c.font = Font(bold=True)
    ws.freeze_panes = "A2"


# =============================================================================
# 17. Batch driver
# =============================================================================

import re

SLN_ALLDATA_FOLDER = "/Users/apaalabasak/Desktop/SLN_alldata"
FILE_DATE = "10-23-2024"

ANIMAL_FOLDER_RE = re.compile(r"^R01 Rat (.+), SLN, " + re.escape(FILE_DATE) + r"$")


def discover_animal_folders(root):
    animals = []
    if not os.path.isdir(root):
        print(f"ERROR: SLN_alldata folder not found: {root}")
        return animals
    for entry in sorted(os.listdir(root)):
        full_path = os.path.join(root, entry)
        if not os.path.isdir(full_path):
            continue
        m = ANIMAL_FOLDER_RE.match(entry)
        if m:
            animals.append((m.group(1), full_path))
        else:
            print(f"Skipping folder (doesn't match expected 'R01 Rat <id>, SLN, {FILE_DATE}' pattern): {entry}")
    return animals


def main():
    animals = discover_animal_folders(SLN_ALLDATA_FOLDER)
    print(f"Found {len(animals)} animal folder(s) to process in: {SLN_ALLDATA_FOLDER}\n")

    pooled_sg, pooled_latency, pooled_intensity, pooled_apnea, pooled_baseline = [], [], [], [], []
    pooled_excluded, pooled_episodes = [], []
    pooled_lags, pooled_sbc = [], []
    processed_animals, skipped_animals = [], []

    for animal_id, folder in animals:
        print(f"{'=' * 60}\nProcessing Rat {animal_id}...\n{'=' * 60}")
        file_prefix = f"R01 Rat {animal_id}, SLN, {FILE_DATE}, "
        try:
            result = process_animal(folder, file_prefix, animal_id)
        except Exception:
            import traceback
            print(f"  ERROR: Rat {animal_id} crashed during processing -- SKIPPING this animal "
                  f"so the pooled summary is still written. Details:")
            traceback.print_exc()
            plt.close("all")
            result = None
        if result is None:
            skipped_animals.append(animal_id)
            continue
        sg_df, lat_df, intensity_df, apnea_df, baseline_df, excluded_df, episodes_df, lags_df, sbc_df = result
        pooled_sg.append(sg_df)
        pooled_latency.append(lat_df)
        pooled_intensity.append(intensity_df)
        pooled_apnea.append(apnea_df)
        pooled_baseline.append(baseline_df)
        pooled_excluded.append(excluded_df)
        pooled_episodes.append(episodes_df)
        pooled_lags.append(lags_df)
        pooled_sbc.append(sbc_df)
        processed_animals.append(animal_id)
        print()

    print(f"{'=' * 60}")
    print(f"Processed {len(processed_animals)} animal(s): {processed_animals}")
    if skipped_animals:
        print(f"SKIPPED {len(skipped_animals)} animal(s) due to missing/incomplete data: {skipped_animals}")
    print(f"{'=' * 60}\n")

    if not processed_animals:
        print("No animals were successfully processed -- pooled summary not created.")
        return

    sg_all = pd.concat(pooled_sg, ignore_index=True)
    latency_all = pd.concat(pooled_latency, ignore_index=True)
    intensity_all = pd.concat(pooled_intensity, ignore_index=True)
    apnea_all = pd.concat(pooled_apnea, ignore_index=True)
    baseline_all = pd.concat(pooled_baseline, ignore_index=True)
    excluded_all = pd.concat(pooled_excluded, ignore_index=True)
    episodes_all = pd.concat(pooled_episodes, ignore_index=True)
    lags_all = pd.concat(pooled_lags, ignore_index=True)
    sbc_all = pd.concat(pooled_sbc, ignore_index=True)

    CONDITION_ORDER = ["A40", "D40", "A30", "D30"]

    def _pivot(df, value_col, block_label, condition_col="Condition"):
        if "Block" in df.columns:
            sub = df[df["Block"] == block_label]
        else:
            sub = df
        if sub.empty or value_col not in sub.columns:
            return None
        pivot = sub.pivot_table(index="Animal", columns=condition_col, values=value_col, aggfunc="first")
        pivot = pivot.reindex(columns=[c for c in CONDITION_ORDER if c in pivot.columns] or CONDITION_ORDER)
        return pivot

    PARAMETERS = [
        ("Max_SG_Amp_Top10",        sg_all,       "Max_SG_Amplitude_g",   "g"),
        ("Avg_SG_Amp_Top10",        sg_all,       "Avg_SG_Amplitude_g",   "g"),
        ("Max_TongueForce_Top10",   sg_all,       "Max_Tongue_Force_N",   "N"),
        ("Avg_TongueForce_Top10",   sg_all,       "Avg_Tongue_Force_N",   "N"),
        ("Max_SG_Amp_WholeTrain",   sg_all,       "Max_SG_Amplitude_WholeTrain_g", "g"),
        ("Avg_SG_Amp_WholeTrain",   sg_all,       "Avg_SG_Amplitude_WholeTrain_g", "g"),
        ("Max_TF_WholeTrain",       sg_all,       "Max_Tongue_Force_WholeTrain_N", "N"),
        ("Avg_TF_WholeTrain",       sg_all,       "Avg_Tongue_Force_WholeTrain_N", "N"),
        ("Swallow_Count",           latency_all,  "N_swallows_in_stim_window", "#"),
        ("Swallow_Count_Full20s",   latency_all,  "N_swallows_full_20sec", "#"),
        ("Max_Peak_ExcludedWindow",      latency_all,  "Max_Peak_In_Excluded_Window_g", "g"),
        ("Swallow_Latency",         latency_all,  "Swallow_latency_sec",  "s"),
        ("Messy_Signal_Duration",   latency_all,  "Messy_Signal_Duration_sec", "s"),
        ("Messy_Signal_Uncapped",   latency_all,  "Messy_Signal_Duration_Uncapped_ms", "ms"),
        ("Swallow_Rate",            latency_all,  "Swallow_Rate_Hz",      "Hz"),
        ("Inter_Swallow_Interval",  latency_all,  "Inter_Swallow_Interval_ms", "ms"),
        ("SwRR_Apnea_Rate_Assumed", apnea_all,    f"{APNEA_LABEL_SWALLOW_RR}_rate_assumed", "#"),
        ("SwRR_Apnea_Rate_Verified", apnea_all,   f"{APNEA_LABEL_SWALLOW_RR}_rate_verified", "#"),
        ("SwRR_Apnea_Latency",      apnea_all,    f"{APNEA_LABEL_SWALLOW_RR}_first_latency_sec", "s"),
        ("SwRR_Apnea_Duration",     apnea_all,    f"{APNEA_LABEL_SWALLOW_RR}_mean_duration_sec", "s"),
        ("StimRR_Apnea_Duration",   apnea_all,    f"{APNEA_LABEL_STIM_RR}_duration_sec", "s"),
        ("Extended_Apnea_Duration", apnea_all,    "Extended_Apnea_Duration_sec", "s"),
        ("Swallows_During_Apnea",   apnea_all,    "N_Swallows_During_Apnea", "#"),
        ("StimRR_Apnea_Verified",   apnea_all,    f"{APNEA_LABEL_STIM_RR}_verified", None),
    ]

    # Baseline sheets added on top of the V7 pooled workbook
    BASELINE_PARAMETERS = [
        ("SG_Baseline_Value",       sg_all,       "Baseline_Value_g",            "g"),
        ("SG_Baseline_PeaksExcl",   sg_all,       "Baseline_Peaks_Excluded",     "#"),
        ("SG_Baseline_Corrected",   sg_all,       "Baseline_Correction_Applied", None),
        ("SG_Baseline_SamplesUsed", baseline_all, "PreStim_Samples_Used_pct",    "%"),
    ]

    # Extended-apnea sheets added on top of the V7 pooled workbook
    EXT_APNEA_PARAMETERS = [
        ("ExtApnea_N_Episodes",     apnea_all,    "N_Extended_Apnea_Episodes",   "#"),
        ("ExtApnea_Total_Duration", apnea_all,    "Total_Apnea_Duration_sec",    "s"),
        ("Breathing_Duration",      apnea_all,    "Breathing_Duration_sec",      "s"),
        ("ExtApnea_Swallow_Rate",   apnea_all,    "Swallow_Rate_During_Apnea_Hz", "Hz"),
        # Swallow-breath lag (swallow SG peak - nearest RR peak)
        ("SBC_N_Measured",          sbc_all,      "N_Swallows_Lag_Measured",     "#"),
        ("SBC_Mean",                sbc_all,      "Swallow_RR_Lag_mean_ms",      "ms"),
        ("SBC_Median",              sbc_all,      "Swallow_RR_Lag_median_ms",    "ms"),
        ("SBC_N_Positive",          sbc_all,      "N_Lag_Positive",              "#"),
        ("SBC_Pct_Positive",        sbc_all,      "Pct_Lag_Positive",            "%"),
        ("SBC_N_Negative",          sbc_all,      "N_Lag_Negative",              "#"),
        ("SBC_Pct_Negative",        sbc_all,      "Pct_Lag_Negative",            "%"),
        ("SBC_N_Zero",              sbc_all,      "N_Lag_Zero",                  "#"),
        ("SBC_N_Outside_Cycle",     sbc_all,      "N_Swallows_Outside_Resp_Cycle", "#"),
        ("SBC_Positive_Values",     sbc_all,      "Lag_Positive_values_ms",      "ms"),
        ("SBC_Positive_Mean",       sbc_all,      "Lag_Positive_mean_ms",        "ms"),
        ("SBC_Positive_Median",     sbc_all,      "Lag_Positive_median_ms",      "ms"),
        ("SBC_Negative_Values",     sbc_all,      "Lag_Negative_values_ms",      "ms"),
        ("SBC_Negative_Mean",       sbc_all,      "Lag_Negative_mean_ms",        "ms"),
        ("SBC_Negative_Median",     sbc_all,      "Lag_Negative_median_ms",      "ms"),
    ]

    pooled_path = os.path.join(SLN_ALLDATA_FOLDER, "pooled_analysis_summary.xlsx")
    excl_sheet = f"Excluded Window (first {TIME_EXCLUDE:g}s)"
    sheet_of = {id(sg_all): "SG Summary (Amp & Force)", id(latency_all): "Swallow Counts & Latency",
                id(apnea_all): "Apnea Summary", id(baseline_all): "Baseline Correction (SG)", id(sbc_all): SBC_SHEET,
                id(excluded_all): excl_sheet}
    readme = pooled_readme_df(
        [(PARAMETERS, sheet_of), (BASELINE_PARAMETERS, sheet_of), (EXT_APNEA_PARAMETERS, sheet_of)],
        [("Max_Intensity", "Max Intensity", "Strongest-force condition per animal and block."),
         ("Baseline_Correction_SG", "Baseline Correction (SG)", "SG baseline details for every animal."),
         ("Extended_Apnea_Episodes_All", "Extended Apnea Episodes", "Every extended apnea episode of every animal."),
         ("SBC_Values_All", SBC_VALUES_SHEET,
          "Every Swallow-Breathing Coordination (SBC) value of every animal, in ms: one row per paired swallow "
          "(one per star on the RR plots). Positive values in SBC_Positive (ms), negative values in SBC_Negative (ms), "
          "each keeping its sign."),
         (f"Excluded_Window_{TIME_EXCLUDE:g}s_All"[:31], excl_sheet,
          f"First {TIME_EXCLUDE:g} s results for every animal.")])

    with pd.ExcelWriter(pooled_path, engine="openpyxl") as writer:
        sheets_written = 0
        readme.to_excel(writer, sheet_name=README_SHEET_NAME, index=False)
        def _write_pivots(param_list):
            n = 0
            for name_prefix, source_df, value_col, unit in param_list:
                for block_label, block_suffix in (("Block 1", "B1"), ("Block 2", "B2")):
                    pivot = _pivot(source_df, value_col, block_label)
                    if pivot is None:
                        continue
                    base_name = f"{name_prefix}_{block_suffix}"
                    unit_tag = f" ({unit})" if unit else ""
                    sheet_name = (base_name + unit_tag)[:31]
                    pivot.to_excel(writer, sheet_name=sheet_name, index=True)
                    n += 1
            return n

        sheets_written += _write_pivots(PARAMETERS)

        if not intensity_all.empty:
            intensity_all_renamed = intensity_all.rename(columns={
                "Intensity_uA": "Intensity (uA)",
                "Max_Tongue_Force_N": "Max_Tongue_Force (N)",
            })
            intensity_all_renamed.to_excel(writer, sheet_name="Max_Intensity", index=False)
            sheets_written += 1

        if not baseline_all.empty:
            baseline_all.to_excel(writer, sheet_name="Baseline_Correction_SG", index=False)
            sheets_written += 1

        sheets_written += _write_pivots(BASELINE_PARAMETERS)

        sheets_written += _write_pivots(EXT_APNEA_PARAMETERS)

        if not episodes_all.empty:
            episodes_all.to_excel(writer, sheet_name="Extended_Apnea_Episodes_All", index=False)
            sheets_written += 1

        if not lags_all.empty:
            lags_all.rename(columns={"Swallow_time_sec": "Swallow_time (s)",
                                     "Resp_cycle_RR_peak_time_sec": "Resp_cycle_RR_peak_time (s) (T = 0)",
                                     "SBC_Positive_ms": "SBC_Positive (ms)", "SBC_Negative_ms": "SBC_Negative (ms)"}
                            ).to_excel(writer, sheet_name="SBC_Values_All", index=False)
            sheets_written += 1

        if not excluded_all.empty:
            excluded_all.to_excel(writer, sheet_name=f"Excluded_Window_{TIME_EXCLUDE:g}s_All"[:31],
                                  index=False)
            sheets_written += 1

        for sheet_name, sheet in writer.sheets.items():
            if sheet_name == README_SHEET_NAME:
                continue
            for col_cells in sheet.columns:
                length = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
                sheet.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 10), 40)
        format_readme_sheet(writer.sheets[README_SHEET_NAME], [44, 45, 55, 100])

    print(f"Wrote pooled cross-animal summary ({sheets_written} sheets) to: {os.path.abspath(pooled_path)}")
    print(f"  Animals included in pooled file: {processed_animals}")
    if skipped_animals:
        print(f"  Animals NOT included (see warnings/errors above): {skipped_animals}")


if __name__ == "__main__":
    main()