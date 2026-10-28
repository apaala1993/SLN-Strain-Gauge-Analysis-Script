# SLN-Strain-Gauge-Analysis-Script
Python pipeline for analyzing tongue strain-gauge and respiration recordings during SLN-evoked swallowing in rats. It is used to test whether tongue exercise improves swallowing tongue force after hypoglossal (XII) motor neuron loss (CTB-SAP model).

**SLN Swallow and Breathing Analysis**

Analysis code for a study testing whether tongue exercise increases tongue force during swallowing in a rat model of hypoglossal (XII) motor neuron loss (intralingual CTB-SAP). Swallows are evoked by superior laryngeal nerve (SLN) electrical stimulation, and the code measures swallowing, tongue force and breathing from each recording, with figures that mark every detection for visual checking.

What the code measures

The code analyzes every stimulus train at 30 and 40 µA in both series (A30, A40, D30, D40), for both repeats (Block 1 and Block 2). The three baseline recordings (no stimulation) are used only as a quality check.

1. Finding each stimulation trial

The comments typed in LabChart during the experiment mark each part of the protocol. The code uses them to find each condition, then finds the exact stimulus onset and offset (about 20 s apart) as the first and last points where the stimulus channel crosses a fixed threshold. Each trial is cut from 5 s before stimulus onset; all swallow and breathing measures use the period between onset and offset.

2. Tongue force baseline

The resting level of the strain gauge differs between trials. For each trial the code measures the baseline from the 5 s before stimulus onset, ignoring any spontaneous peaks there, and takes the median of what remains. This value is subtracted from the whole trial, so every tongue force is measured from the animal's resting level just before stimulation.

3. Swallows

Swallows are the peaks in the baseline-corrected strain-gauge signal.

Automatic threshold. Instead of one fixed cut-off for all animals, the minimum peak size is set for each trial with Otsu's method, which splits the peak sizes into large peaks (swallows) and small fluctuations.
Peak size = prominence: how far a peak rises above the lowest points on either side of it, so slow drifts in the signal do not affect detection.
Peaks closer together than about one third of the typical swallow interval, and secondary bumps on the falling edge of a swallow that never return toward baseline, count as one swallow.
Only swallows between stimulus onset and offset are used.
The first 2 s after onset often contain stimulation artifact. Swallows there are reported on their own and left out of the main swallow outcomes.

For each trial the code reports:

Measure	What it is:

Swallow count	Number of swallows after the first 2 s
Swallow rate	Swallows per second over the stimulation period after the first 2 s
Inter-swallow interval	Mean time between consecutive swallows
Swallow latency	Time from stimulus onset to the first swallow (including the first 2 s)
Maximum and mean tongue force	From the 10 largest swallows, converted from grams to newtons (0.332 N per g)
Whole-train force	Maximum and mean of the signal over the entire stimulus train
Strongest condition	The condition (A30, A40, D30, D40) giving the highest tongue force in each block

4. Breaths

Breaths are the peaks in the respiration signal, judged against each animal's own breathing:

The breaths in the 5 s before stimulus onset define that animal's normal breath size (median peak prominence) and normal breathing interval (median time between breaths).
During stimulation, a peak counts as a breath only if it is at least half the normal breath size.
Smaller deflections, 25 to 50% of normal breath size, are blips: not breaths, left out of every breathing measure, and marked with a grey X on the figures. Anything below 25% is treated as noise.

Because breaths are judged relative to each animal's own pre-stimulus breathing, the same rules work across animals with different signal sizes.

5. Apnea (pauses in breathing)
Swallow-related apnea: time from each swallow to the next breath.
Stimulus-onset apnea: time from stimulus onset to the first breath.

Each is called a true (verified) apnea when it is longer than 1.5 times the local normal breathing interval. This allows for normal breath-to-breath variability while flagging pauses long enough that roughly one breath was missed.

6. Extended apnea

An extended apnea is a stretch of the stimulation period without normal breathing that lasts at least three normal breathing intervals, with a 1.5 s minimum. Scaling with the animal's breathing rate makes the rule fair across animals, and the minimum stops brief pauses in fast-breathing animals being counted.

An episode ends only when normal breathing resumes: two breaths within that interval of each other. One isolated breath does not end an episode.
An animal can have several episodes during one stimulus train.

For each trial the code reports:

the number of episodes;
the duration of each episode;
the number and rate of swallows during each episode;
the total time in extended apnea (sum of all episodes);
the time spent breathing (stimulus duration minus total apnea).
Outputs

For each animal, in a figures_output folder inside the animal's data folder:

analysis_summary.xlsx: all measures for that animal, one sheet per topic. The first sheet, READ ME - Definitions, explains every column and lists the settings used for that run.
PNG/ and SVG/: the same figures in both formats; the SVGs can be edited in Illustrator or Inkscape. The figures mark every detected swallow, the 10 swallows used for force, the baseline, breaths, rejected blips and extended-apnea boundaries.

Across all animals, in the main data folder:

pooled_analysis_summary.xlsx: one sheet per measure, with animals as rows and the four conditions (A40, D40, A30, D30) as columns, separately for Block 1 and Block 2. Each cell is that animal's own value; nothing is averaged.
How the results were checked
Visual review. For every trial and every animal, the figures marking each detected swallow, the baseline, rejected respiratory deflections and apnea-episode boundaries were reviewed against the raw signals. The thresholds separating breaths from blips were chosen by inspecting the recordings and then applied to all animals.
Comparison with manual counts. Results generated by the code were compared with values counted manually, for both the individual-animal and the pooled summaries, with no discrepancies found.
Simulated recordings with known answers. The code recovered known baseline offsets (2 g and 25 g) to within 0.01 g while ignoring a spontaneous pre-stimulus peak, found the known swallows and breathing pauses, and rejected deflections that were not breaths.
Internal consistency. Summary values were recalculated independently from the individual detected events, and the pooled file was cross-checked against each animal's file.
Running the code

Requirements: Python 3.11 with NumPy, SciPy, pandas, Matplotlib and openpyxl. For example, with conda:

conda create -n sln python=3.11 numpy scipy pandas matplotlib openpyxl -y

Data layout. Each animal has its own folder inside one main folder, named exactly R01 Rat <ID>, SLN, <date>, containing the four LabChart exports:

SLN_alldata/
  R01 Rat X, SLN, 10-23-2024/
    R01 Rat X, SLN, 10-23-2024, Ch1 RR.mat
    R01 Rat X, SLN, 10-23-2024, Ch2 STIM.mat
    R01 Rat X, SLN, 10-23-2024, Ch3 SG.mat
    R01 Rat X, SLN, 10-23-2024, Ch9 EMG RMS.mat

Run. Set SLN_ALLDATA_FOLDER and FILE_DATE near the bottom of the script, then run it. Every animal in the folder is analyzed, followed by the pooled summary.

Settings. All adjustable values are at the top of the script, each with a comment explaining it. The main ones:

**Setting	Default	Controls**

TIME_EXCLUDE	2.0 s	Length of the excluded period at the start of stimulation
NUMBER_PEAK_INCLUDE	10	Number of largest swallows used for tongue force (0 = all)
calibration_factor	0.332 N/g	Strain-gauge calibration
BREATH_MIN_PROMINENCE_FRACTION	0.5	Minimum breath size, relative to a normal breath
BLIP_MIN_PROMINENCE_FRACTION	0.25	Smallest deflection marked as a blip
EXTENDED_APNEA_IBI_MULTIPLIER	3.0	Extended apnea length, in normal breathing intervals
EXTENDED_APNEA_MIN_SEC	1.5 s	Minimum extended apnea length
VERIFIED_APNEA_MULTIPLIER	1.5	Threshold for a verified swallow or stimulus-onset apnea
FIGURE_DPI	800	PNG resolution
REMOVE_FIGURE_TEXT	True	Text-free figures (only swallow numbers kept) for publication

