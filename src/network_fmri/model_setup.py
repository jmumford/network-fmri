"""Build first-level design matrices for fsLR den-91k data.

Every regressor is defined the same way, in the task's YAML file:

    <name>:
      amplitude: 1 | <events column>
      duration:  duration | response_time | <number>
      subset:    <pandas query against the events file> | null

There are no derived columns and no task-specific logic in this module. Every
name a YAML mentions is a column that exists in the events.tsv, so you can open
the events file and the YAML side by side and see exactly what the model is.

Adding a task means writing one YAML. Nothing here needs to change; a task
without a YAML raises.

DATA ASSUMPTIONS
----------------
Written for the untrimmed fMRIPrep derivatives, e.g.

    /oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_20250402
    .../derivatives/fmriprep_25.2.4

For those, three corrections are applied (confirmed with the pipeline's author):

  1. the first ``dummy_scans`` (7) volumes are dropped from the dtseries;
  2. event onsets are shifted by ``-dummy_scans * TR`` (-10.43 s), because the
     design's time axis restarts at zero on the first retained volume;
  3. the confounds TSV is trimmed by the same 7 rows, since fMRIPrep wrote one
     row per untrimmed volume.

Nothing in the files records whether this has already been done, so
``data_is_trimmed`` is a required argument with no default. When the canonical
data become pre-trimmed, delete that argument and the three corrections
outright rather than leaving an option nobody can resolve from the data.

TR comes from the fMRIPrep sidecar and is cross-checked against the task
YAML's ``tr``; a disagreement raises. The warmup count comes from the YAML's
``dummy_scans``. Neither is hardcoded here.
"""

import contextlib
import io
import json
import keyword
import logging
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from nilearn.glm.first_level import make_first_level_design_matrix

logger = logging.getLogger(__name__)

# HRF model, passed to nilearn. No derivatives and no drift terms: high-pass
# filtering comes from the cosine basis in the confounds.
HRF_MODEL = "spm"

# ---------------------------------------------------------------------------
# Confound sets
#
# Names are fMRIPrep confounds-TSV columns. An entry ending in ``*`` is globbed
# at load time, which ``cosine*`` needs: fMRIPrep emits a run-length-dependent
# number of DCT terms, so the count varies and cannot be listed.
# ---------------------------------------------------------------------------

MOTION_6 = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]

# Friston-24: the 6 rigid-body parameters, their temporal derivatives, and the
# squares of both.
MOTION_24 = MOTION_6 + [
    f"{p}{suffix}"
    for p in MOTION_6
    for suffix in ("_derivative1", "_power2", "_derivative1_power2")
]

TISSUE = ["csf", "white_matter"]

# fMRIPrep's discrete cosine basis. Including these IS the high-pass filter --
# do not also pass a high_pass or drift_model to the GLM, or the data get
# filtered twice. Exclude them and you need one of those instead.
COSINE = ["cosine*"]

DEFAULT_CONFOUNDS = MOTION_24 + TISSUE + COSINE


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_run_files(bids_dir, fmriprep_dir, subject, task, session, run):
    """Locate the files needed to build a design matrix for one run (fsLR).

    Args:
        bids_dir: BIDS dataset root.
        fmriprep_dir: fMRIPrep derivatives root.
        subject: Subject label, with or without the 'sub-' prefix.
        task: Task name (e.g. 'goNogo').
        session: Session entity, including prefix (e.g. 'ses-01').
        run: Run entity, including prefix (e.g. 'run-1').

    Returns:
        Dict with keys 'events', 'confounds', 'dtseries', 'sidecar'.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    bids_dir = Path(bids_dir)
    fmriprep_dir = Path(fmriprep_dir)
    sub = subject if str(subject).startswith("sub-") else f"sub-{subject}"
    stem = f"{sub}_{session}_task-{task}_{run}"

    fmriprep_func = fmriprep_dir / sub / session / "func"
    bids_func = bids_dir / sub / session / "func"

    run_files = {
        "events": bids_func / f"{stem}_events.tsv",
        "confounds": fmriprep_func / f"{stem}_desc-confounds_timeseries.tsv",
        "dtseries": fmriprep_func / f"{stem}_space-fsLR_den-91k_bold.dtseries.nii",
    }

    # StartTime/DelayTime are set during native-space slice-timing correction
    # and copied identically into every output space's sidecar, so any will do.
    # fsLR preferred for consistency with the dtseries being loaded.
    for pattern in (
        f"{stem}_space-fsLR_den-91k_bold.json",
        f"{stem}_space-T1w_desc-preproc_bold.json",
        f"{stem}_space-*_desc-preproc_bold.json",
    ):
        matches = sorted(fmriprep_func.glob(pattern))
        if matches:
            run_files["sidecar"] = matches[0]
            break

    missing = [k for k, p in run_files.items() if not p.exists()]
    if "sidecar" not in run_files:
        missing.append("sidecar")
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(sorted(missing))} for {stem} "
            f"(looked in {fmriprep_func} and {bids_func})"
        )

    return run_files


def get_timing(sidecar, expected_tr=None):
    """Return (tr, start_time) in seconds, read from fMRIPrep's sidecar.

    The sidecar is the only source of TR used for modelling. Nothing here falls
    back to a hardcoded value: a TR carried over from another dataset produces
    a design matrix that is wrong by a factor, with no error and no obvious
    symptom.

    ``expected_tr`` (normally the task YAML's ``tr``) is checked against the
    sidecar and any disagreement raises -- the cheap guard against pointing a
    task config at a dataset it was not written for.

    ``start_time`` is fMRIPrep's ``StartTime``: the slice-timing reference the
    data were corrected to, in SECONDS. It is added directly to ``frame_times``
    -- it is NOT the fraction-of-TR that ``FirstLevelModel(slice_time_ref=...)``
    expects. Here it is 0.701 s, the midpoint of the true acquisition window
    (TR 1.49 - DelayTime 0.08765), not the naive TR/2 = 0.745.

    Unaffected by volume trimming: it is a within-TR offset.
    """
    meta = json.loads(Path(sidecar).read_text())
    if "RepetitionTime" not in meta:
        raise ValueError(
            f"{sidecar} has no RepetitionTime. Check for a task-level sidecar "
            f"higher in the BIDS tree (RepetitionTime may be inherited)."
        )
    tr = float(meta["RepetitionTime"])

    if expected_tr is not None and not np.isclose(tr, float(expected_tr), atol=1e-4):
        raise ValueError(
            f"TR mismatch: sidecar {Path(sidecar).name} says {tr}s, but the task "
            f"config says {expected_tr}s. The task config was probably written "
            f"for a different dataset. Fix the config (or pass expected_tr=None) "
            f"rather than proceeding."
        )

    start_time = meta.get("StartTime")
    if start_time is None:
        logger.warning(
            "No StartTime in %s; assuming 0.0. If these data were slice-time "
            "corrected, regressors will be misaligned by up to TR/2.",
            sidecar,
        )
        return tr, 0.0
    return tr, float(start_time)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_bold(dtseries, n_dummy, dtype=np.float32, load_data=True):
    """Load a dtseries and drop the first ``n_dummy`` volumes.

    Args:
        dtseries: Path to a ``*_space-fsLR_den-91k_bold.dtseries.nii``.
        n_dummy: Volumes to drop from the start.
        dtype: Output dtype. float32 halves memory vs nibabel's float64.
        load_data: When False, read only the header and return
            ``(None, n_scans)``. A design matrix needs nothing from the
            timeseries but its length, so this saves ~120 MB and a second or
            two per run when looping over a study.

    Returns:
        (data, n_scans). ``n_scans`` is the TRIMMED length -- the value that
        must drive frame_times.

    There is no way to tell from the file whether the warmup volumes are still
    present: nothing in the CIFTI header, the sidecar, or the confounds records
    it, and fMRIPrep does not populate BIDS's
    ``NumberOfVolumesDiscardedByUser``. The caller's ``data_is_trimmed`` is the
    only thing preventing a double-applied correction.
    """
    img = nib.load(str(dtseries))
    raw_volumes = img.shape[0]  # CIFTI is (time, grayordinates): time is axis 0

    if raw_volumes <= n_dummy:
        raise ValueError(
            f"{Path(dtseries).name}: only {raw_volumes} volumes, cannot drop {n_dummy}"
        )

    n_scans = raw_volumes - n_dummy
    logger.info(
        "%s: %d volumes on disk, %d after dropping %d dummies%s",
        Path(dtseries).name, raw_volumes, n_scans, n_dummy,
        "" if load_data else " (header only)",
    )
    if not load_data:
        return None, n_scans

    # Slice via dataobj so the dummy volumes are never read into memory.
    data = np.asarray(img.dataobj[n_dummy:], dtype=dtype)
    return data, data.shape[0]


# Columns coerced to numeric on load. A single stray non-numeric string would
# otherwise leave the column as dtype object and make every comparison in a
# subset silently False.
_NUMERIC_EVENT_COLUMNS = (
    "onset", "duration", "response_time", "key_press", "correct_response", "acc",
)


def load_events(events_tsv, tr, n_dummy, n_scans=None, drop_negative_onsets=False):
    """Load a BIDS events.tsv, shift onsets, and coerce numeric columns.

    This is the only processing applied to the events. No columns are added and
    no rows are reclassified -- what the subsets in the YAML see is what is in
    the file.

    Args:
        events_tsv: Path to the events file.
        tr: Repetition time, for the onset shift.
        n_dummy: Warmup volumes removed from the BOLD; onsets shift by
            ``-n_dummy * tr`` to match.
        n_scans: Trimmed BOLD length. When given, rows with
            ``onset >= n_scans * tr`` are dropped -- they describe trials that
            were never imaged (an aborted run leaves the behavioural session
            running).
        drop_negative_onsets: Default False. An event a few seconds before the
            first retained volume still contributes its HRF to the retained
            volumes, so dropping it discards real signal and leaves that
            variance unmodelled. Set True only to match a pipeline that
            dropped them.

    Returns:
        Events DataFrame.
    """
    events = pd.read_csv(events_tsv, sep="\t", na_values=["n/a"])
    if "onset" not in events.columns:
        raise ValueError(f"{events_tsv} has no 'onset' column")

    for col in _NUMERIC_EVENT_COLUMNS:
        if col in events.columns:
            events[col] = pd.to_numeric(events[col], errors="coerce")

    shift = n_dummy * tr
    events["onset"] = events["onset"] - shift

    n_negative = int((events["onset"] < 0).sum())
    if n_negative and drop_negative_onsets:
        events = events[events["onset"] >= 0].copy()
        logger.warning(
            "Shifted onsets by -%.4fs and DROPPED %d event(s) with negative onsets",
            shift, n_negative,
        )
    elif n_negative:
        logger.info(
            "Shifted onsets by -%.4fs; %d event(s) now have negative onsets (kept)",
            shift, n_negative,
        )

    if n_scans is not None:
        bold_duration = n_scans * tr
        before = len(events)
        events = events[events["onset"] < bold_duration].copy()
        if before - len(events):
            logger.info(
                "Dropped %d event(s) starting after the BOLD ends (%.1fs)",
                before - len(events), bold_duration,
            )

    # A negative response_time means "no response" in some conversions and is a
    # trap for subsets: `response_time < 0.2` would be True for it. NaN is safe
    # (all comparisons are False), so warn rather than transform silently.
    if "response_time" in events.columns and (events["response_time"] < 0).any():
        n_neg = int((events["response_time"] < 0).sum())
        logger.warning(
            "%d event(s) have a negative response_time. Subsets testing "
            "`response_time < X` will match them -- write "
            "`response_time >= 0 and response_time < X` instead.",
            n_neg,
        )

    return events


def load_confounds(confounds_tsv, columns, n_scans, n_dummy):
    """Load selected confound columns, trimmed to match the BOLD.

    fMRIPrep wrote one row per UNtrimmed volume, so the same ``n_dummy`` rows
    come off here. The index is reset because the design matrix is assembled
    with ``pd.concat(axis=1)``, which aligns on index: a sliced frame still
    carrying RangeIndex(n_dummy, raw) would union with the regressors'
    RangeIndex(0, n_scans) and produce an all-NaN matrix.

    Args:
        confounds_tsv: Path to ``*_desc-confounds_timeseries.tsv``.
        columns: Column names. An entry ending in ``*`` is globbed, which
            ``cosine*`` needs. A literal name that is absent raises (a typo);
            a wildcard matching nothing only warns (legitimate on a short run).
        n_scans: Trimmed BOLD length, for the consistency check.
        n_dummy: Rows to drop from the start.

    Returns:
        DataFrame of shape (n_scans, n_matched), RangeIndex from 0.
    """
    confounds = pd.read_csv(confounds_tsv, sep="\t", na_values=["n/a"])

    if n_dummy and len(confounds) != n_scans + n_dummy:
        raise ValueError(
            f"{Path(confounds_tsv).name}: {len(confounds)} rows, expected "
            f"{n_scans + n_dummy} (n_scans {n_scans} + {n_dummy} dummies). "
            f"Are the BOLD and confounds from the same preprocessing run?"
        )

    resolved, missing = [], []
    for spec in columns:
        if spec.endswith("*"):
            hits = sorted(c for c in confounds.columns if c.startswith(spec[:-1]))
            if not hits:
                logger.warning(
                    "No confound columns match %r in %s", spec, Path(confounds_tsv).name
                )
            resolved.extend(h for h in hits if h not in resolved)
        elif spec in confounds.columns:
            if spec not in resolved:
                resolved.append(spec)
        else:
            missing.append(spec)
    if missing:
        raise ValueError(
            f"Confound column(s) not found in {Path(confounds_tsv).name}: {missing}"
        )

    trimmed = confounds.iloc[n_dummy:][resolved].reset_index(drop=True)

    # fMRIPrep's first-volume regressors (non_steady_state_outlier*, and any
    # motion_outlier flagging a trimmed frame) go all-zero after the trim and
    # would be zero-variance columns in the design.
    all_zero = [c for c in trimmed.columns if (trimmed[c] == 0).all()]
    if all_zero:
        logger.warning("Confound column(s) all zero after trimming, dropping: %s",
                       all_zero)
        trimmed = trimmed.drop(columns=all_zero)

    n_nan = int(trimmed.isna().sum().sum())
    if n_nan:
        logger.warning("Filling %d NaN(s) in confounds with 0", n_nan)
        trimmed = trimmed.fillna(0)

    if len(trimmed) != n_scans:
        raise ValueError(f"Confounds length {len(trimmed)} != n_scans {n_scans}")
    return trimmed


def build_frame_times(n_scans, tr, start_time):
    """Sampling times of the retained volumes, in seconds.

    ``np.arange(n_scans) * tr + start_time``. Volume k is treated as sampling
    the signal at ``k * tr + start_time``, the slice-timing reference the data
    were corrected to. Omitting ``start_time`` would shift every regressor
    0.701 s early -- small, but systematic.
    """
    return np.arange(n_scans) * tr + start_time


def compute_global_signal(bold, structures=None, axis=None):
    """Mean grayordinate timeseries -- the fsLR analogue of global signal.

    The ``global_signal`` column in fMRIPrep's confounds TSV is a mean over the
    VOLUMETRIC brain mask, which is not the space being modelled, so it is the
    wrong regressor here. This averages the grayordinates actually being fit.

    Args:
        bold: (n_scans, n_grayordinates), already trimmed. Trimming must
            precede the mean or the warmup volumes -- whose intensity has not
            reached steady state -- shift it.
        structures: Optional CIFTI structure names to restrict to, e.g. the two
            cortex structures. Requires ``axis``. Default averages all 91,282
            grayordinates, including subcortical.
        axis: The dtseries' BrainModelAxis (axis 1), needed for ``structures``.

    Returns:
        1-D array of length n_scans.
    """
    if bold is None:
        raise ValueError(
            "compute_global_signal needs the timeseries; call "
            "build_run_design_matrix with global_signal=True (which forces the "
            "data load) rather than load_data=False."
        )

    if structures:
        if axis is None:
            raise ValueError("structures= requires the dtseries BrainModelAxis")
        names = np.asarray([str(n) for n in axis.name])
        keep = np.isin(names, [str(s) for s in structures])
        if not keep.any():
            raise ValueError(
                f"No grayordinates matched {structures}. "
                f"Available: {sorted(set(names))}"
            )
        logger.info("Global signal over %d of %d grayordinates (%s)",
                    int(keep.sum()), bold.shape[1], ", ".join(structures))
        return bold[:, keep].mean(axis=1)

    return bold.mean(axis=1)


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

_REQUIRED_REGRESSOR_FIELDS = {"amplitude", "duration", "subset"}

# Tokens that appear in a pandas query but are not column names.
_QUERY_NON_COLUMNS = set(keyword.kwlist) | {
    "isna", "notna", "isnull", "notnull", "str", "startswith", "endswith",
    "contains", "abs", "index", "True", "False", "None",
}
_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_STRING_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"")


def load_task_config(task_yaml):
    """Read and validate a task YAML.

    Checks that every regressor has amplitude/duration/subset, and that every
    contrast formula references only declared regressors. Raises on either --
    a contrast naming a regressor that does not exist would otherwise fail much
    later, or silently produce nothing.

    Returns:
        The parsed dict: 'regressors', 'contrasts', and any parameters
        ('tr', 'dummy_scans', ...).
    """
    task_yaml = Path(task_yaml)
    if not task_yaml.exists():
        available = sorted(p.stem for p in task_yaml.parent.glob("*.yaml"))
        raise FileNotFoundError(
            f"No task config at {task_yaml}. "
            f"Available in {task_yaml.parent}: {available}"
        )

    config = yaml.safe_load(task_yaml.read_text()) or {}
    regressors = config.get("regressors") or {}
    contrasts = config.get("contrasts") or {}

    if not regressors:
        raise ValueError(f"{task_yaml.name} defines no regressors")

    for name, cfg in regressors.items():
        if not isinstance(cfg, dict):
            raise ValueError(
                f"{task_yaml.name}: regressor {name!r} must be a mapping with "
                f"amplitude/duration/subset, got {type(cfg).__name__}"
            )
        missing = _REQUIRED_REGRESSOR_FIELDS - set(cfg)
        if missing:
            raise ValueError(
                f"{task_yaml.name}: regressor {name!r} is missing {sorted(missing)}"
            )

    declared = set(regressors)
    for cname, formula in contrasts.items():
        unknown = set(_IDENT_RE.findall(str(formula))) - declared
        if unknown:
            raise ValueError(
                f"{task_yaml.name}: contrast {cname!r} references undeclared "
                f"regressor(s) {sorted(unknown)}. Declared: {sorted(declared)}"
            )

    return config


def referenced_event_columns(config):
    """Every events column a task config needs, as a sorted list.

    Collected from each regressor's subset query and from any amplitude or
    duration given as a column name. Used to check a task config against an
    events file before building anything -- see :func:`check_event_columns`.
    """
    needed = {"onset"}  # always read, never named in the YAML
    for cfg in config["regressors"].values():
        for field in ("amplitude", "duration"):
            value = cfg[field]
            if isinstance(value, str):
                needed.add(value)
        subset = cfg.get("subset")
        if subset:
            # strip string literals so 'test_trial' is not read as a column
            stripped = _STRING_LITERAL_RE.sub(" ", str(subset))
            for tok in _IDENT_RE.findall(stripped):
                if tok not in _QUERY_NON_COLUMNS and not tok.isdigit():
                    needed.add(tok)
    return sorted(needed)


def check_event_columns(config, events, task_name=""):
    """Raise if the task config references a column the events file lacks.

    Runs before any regressor is built, so a typo or a config written for a
    different events layout fails with one clear message instead of partway
    through.
    """
    needed = referenced_event_columns(config)
    missing = [c for c in needed if c not in events.columns]
    if missing:
        raise ValueError(
            f"Task config {task_name} references events column(s) {missing}, "
            f"which are not in the events file.\n"
            f"  needed:    {needed}\n"
            f"  available: {sorted(events.columns)}"
        )


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------


def _resolve(events, value, field, name):
    """Resolve an amplitude/duration spec to a per-event vector.

    A number becomes a constant vector; a string is an events column name.
    """
    if isinstance(value, bool):
        raise ValueError(
            f"Regressor {name!r}: {field} is a boolean; use a number or a "
            f"column name."
        )
    if isinstance(value, (int, float)):
        return np.full(len(events), float(value))
    if value not in events.columns:
        raise ValueError(
            f"Regressor {name!r}: {field} names column {value!r}, which is not "
            f"in the events file. Available: {sorted(events.columns)}"
        )
    return events[value].to_numpy(dtype=float)


def standardize_columns(df, skip=("constant",)):
    """Return a copy of ``df`` with every varying column z-scored.

    Rescaling a regressor cannot change a least-squares fit: the column spans
    the same direction either way, only its length changes, and its coefficient
    absorbs the difference. What it changes is numerical conditioning.

    That matters because fMRIPrep's confounds span about eleven orders of
    magnitude -- ``rot_*_derivative1_power2`` terms sit near 1e-8 while ``csf``
    is above 1000 -- which pushes the design's condition number past 1e10 and
    costs roughly ten of float64's sixteen significant digits in the solve. The
    residuals are still correct (verified against an unscaled fit: they agree
    to six significant figures, and per-grayordinate standard deviations to
    thirteen decimal places), but the margin shrinks as the condition number
    grows, and it varies run to run with the motion parameters. Scaling removes
    the issue instead of requiring it to be monitored.

    Constant columns are skipped -- an intercept has no variance to divide by.

    Note the resulting coefficients are in units of each column's standard
    deviation. Irrelevant for residuals; worth knowing before interpreting a
    confound beta.
    """
    out = df.copy()
    cols = [c for c in out.columns if c not in skip]
    sd = out[cols].std()
    varies = sd[sd > 0].index
    if len(varies):
        out[varies] = (out[varies] - out[varies].mean()) / sd[varies]
    return out


def build_long_events(events, regressors):
    """Turn the YAML regressor definitions into one long-form events frame.

    Each regressor's subset selects rows; those rows become entries with
    ``trial_type`` set to the regressor's name, plus its resolved duration and
    amplitude (nilearn calls the amplitude ``modulation``). One frame for all
    regressors is what ``make_first_level_design_matrix`` consumes.

    Returns:
        (long_events, three_column, empty_regressors)

        ``three_column`` maps each regressor to the
        ``(onsets, durations, amplitudes)`` actually used, for inspection.
        ``empty_regressors`` lists those that selected nothing -- nilearn
        simply omits those columns, so they are reinstated as explicit zero
        columns in :func:`create_design_matrix`.
    """
    frames, three_column, empty = [], {}, []

    for name, cfg in regressors.items():
        subset = events.query(cfg["subset"]) if cfg.get("subset") else events

        if subset.empty:
            logger.warning(
                "Regressor %r selects no events (subset: %s) -- zero regressor. "
                "If this happens in every run, the subset is probably wrong.",
                name, cfg.get("subset"),
            )
            three_column[name] = ([], [], [])
            empty.append(name)
            continue

        onset = subset["onset"].to_numpy(dtype=float)
        duration = _resolve(subset, cfg["duration"], "duration", name)
        modulation = _resolve(subset, cfg["amplitude"], "amplitude", name)

        # NaN produces garbled output, and response_time is NaN on trials with
        # no response.
        valid = ~(np.isnan(onset) | np.isnan(duration) | np.isnan(modulation))
        if not valid.all():
            logger.warning(
                "Regressor %r: dropped %d event(s) with NaN onset/duration/amplitude",
                name, int((~valid).sum()),
            )
            onset, duration, modulation = onset[valid], duration[valid], modulation[valid]

        if len(onset) == 0:
            logger.warning("Regressor %r: no valid events after NaN filter", name)
            three_column[name] = ([], [], [])
            empty.append(name)
            continue

        frames.append(pd.DataFrame({
            "trial_type": name,
            "onset": onset,
            "duration": duration,
            "modulation": modulation,
        }))
        three_column[name] = (onset.tolist(), duration.tolist(), modulation.tolist())

    long_events = (
        pd.concat(frames, ignore_index=True) if frames
        else pd.DataFrame(columns=["trial_type", "onset", "duration", "modulation"])
    )
    return long_events, three_column, empty


def create_design_matrix(events, confounds, regressors, n_scans, frame_times):
    """Assemble task regressors + confounds + intercept.

    One call to nilearn's ``make_first_level_design_matrix`` builds every task
    regressor and adds the intercept. ``drift_model=None`` and
    ``high_pass=None`` because fMRIPrep's ``cosine*`` confounds ARE the
    high-pass filter -- letting nilearn add its own would filter twice.

    Column order is regressors in YAML order, then confounds, then
    ``constant``. (nilearn sorts trial types alphabetically and omits
    regressors with no events, so both are restored here.)

    Note nilearn ignores events starting more than ``min_onset=-24`` seconds
    before the first frame. Onsets shifted by -10.43 s are well inside that.
    """
    long_events, three_column, empty = build_long_events(events, regressors)

    add_regs = add_reg_names = None
    if confounds is not None and not confounds.empty:
        if len(confounds) != n_scans:
            raise ValueError(f"Confounds length {len(confounds)} != n_scans {n_scans}")
        add_regs = confounds.to_numpy()
        add_reg_names = list(confounds.columns)

    # nilearn prints "A 'modulation' column was found..." on every call, with
    # the verbosity hardcoded, so there is no API way to quiet it. Harmless,
    # but over a few hundred runs it buries the warnings that matter.
    with contextlib.redirect_stdout(io.StringIO()):
        design_matrix = make_first_level_design_matrix(
            frame_times,
            events=long_events if len(long_events) else None,
            hrf_model=HRF_MODEL,
            drift_model=None,
            high_pass=None,
            add_regs=add_regs,
            add_reg_names=add_reg_names,
        )

    # Reinstate regressors that selected no events, so a zero-variance column
    # is dropped (and its contrasts skipped) rather than silently missing.
    for name in empty:
        design_matrix[name] = 0.0

    ordered = list(regressors) + [c for c in design_matrix.columns if c not in regressors]
    design_matrix = design_matrix[ordered]

    if design_matrix.isna().any().any():
        bad = design_matrix.columns[design_matrix.isna().any()].tolist()
        raise ValueError(f"Design matrix contains NaN in column(s): {bad}")

    return design_matrix, three_column


def drop_zero_variance(design_matrix):
    """Drop constant columns other than the intercept.

    A regressor whose events are all absent in a run (no omissions, no false
    alarms) is all-zero and would make the design singular. Any contrast
    referencing it is dropped too -- see :func:`filter_contrasts`.
    """
    dropped = [
        c for c in design_matrix.columns
        if c != "constant" and design_matrix[c].nunique() == 1
    ]
    if dropped:
        logger.warning("Dropping zero-variance column(s): %s", dropped)
        design_matrix = design_matrix.drop(columns=dropped)
    return design_matrix, dropped


def filter_contrasts(contrasts, dropped_columns):
    """Drop contrasts whose formula references a dropped regressor."""
    if not dropped_columns:
        return dict(contrasts), []
    dropped_set = set(dropped_columns)
    kept, skipped = {}, []
    for name, formula in contrasts.items():
        if set(_IDENT_RE.findall(str(formula))) & dropped_set:
            skipped.append(name)
        else:
            kept[name] = formula
    if skipped:
        logger.warning("Skipping contrast(s) with dropped regressors: %s", skipped)
    return kept, skipped


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_run_design_matrix(
    bids_dir,
    fmriprep_dir,
    subject,
    task,
    session,
    run,
    task_yaml,
    confound_columns,
    data_is_trimmed,
    drop_negative_onsets=False,
    scale_confounds=True,
    return_bold=False,
    load_data=True,
    global_signal=False,
    global_signal_structures=None,
):
    """Build the design matrix for one run.

    Args:
        bids_dir: BIDS root (for events.tsv).
        fmriprep_dir: fMRIPrep derivatives root.
        subject, task, session, run: Identifiers ('s03', 'goNogo', 'ses-01',
            'run-1').
        task_yaml: Path to the task's YAML.
        confound_columns: fMRIPrep confound names; entries may end in ``*``.
            See MOTION_24, TISSUE, COSINE, DEFAULT_CONFOUNDS.
        data_is_trimmed: REQUIRED, no default. False for the untrimmed OAK
            derivatives: warmup volumes are still present and onsets are raw,
            so both corrections are applied. True if the dtseries has already
            had them removed AND the onsets already shifted.

            This cannot be detected from the files, which is why it has no
            default -- a wrong value produces a silent 10.43 s misalignment
            rather than an error.
        drop_negative_onsets: See :func:`load_events`. Default False (keep).
        return_bold: Also return the trimmed timeseries.
        scale_confounds: Z-score the confound columns, including the global
            signal. Default True. Changes the design's numerical conditioning,
            not the fit or the residuals -- see :func:`standardize_columns`.
            Task regressors are never scaled, so their coefficients stay
            interpretable.
        load_data: When False, read only the dtseries header. The design matrix
            is unaffected -- only its row count comes from the BOLD -- so this
            is the fast path for building or checking many designs. Forced True
            when ``global_signal`` or ``return_bold`` is set.
        global_signal: Append a ``global_signal`` column, the mean over
            grayordinates of the trimmed dtseries. Not the TSV's column, which
            is a volumetric mean. Requires reading the timeseries.
        global_signal_structures: Optional CIFTI structures to restrict the
            mean to.

    Returns:
        Dict with 'design_matrix', 'contrasts', 'dropped_columns',
        'skipped_contrasts', 'three_column', 'n_scans', 'tr', 'start_time',
        'n_dummy', 'events', 'frame_times', 'files', and 'bold' if requested.
    """
    files = find_run_files(bids_dir, fmriprep_dir, subject, task, session, run)

    config = load_task_config(task_yaml)
    tr, start_time = get_timing(files["sidecar"], expected_tr=config.get("tr"))

    if data_is_trimmed:
        n_dummy = 0
    else:
        n_dummy = config.get("dummy_scans")
        if n_dummy is None:
            raise ValueError(
                f"{task_yaml} has no 'dummy_scans' field, so the number of "
                f"warmup volumes to remove is unknown. Add it to the task "
                f"config, or pass data_is_trimmed=True if these data are "
                f"already corrected."
            )
    logger.info(
        "tr=%.4fs start_time=%.4fs n_dummy=%d (data_is_trimmed=%s)",
        tr, start_time, n_dummy, data_is_trimmed,
    )

    need_data = load_data or return_bold or global_signal
    bold, n_scans = load_bold(files["dtseries"], n_dummy=n_dummy, load_data=need_data)

    events = load_events(
        files["events"], tr, n_dummy=n_dummy, n_scans=n_scans,
        drop_negative_onsets=drop_negative_onsets,
    )
    check_event_columns(config, events, task_name=Path(task_yaml).name)

    confounds = load_confounds(
        files["confounds"], confound_columns, n_scans, n_dummy=n_dummy
    )

    if global_signal:
        axis = (nib.load(str(files["dtseries"])).header.get_axis(1)
                if global_signal_structures else None)
        confounds = confounds.copy()
        confounds["global_signal"] = compute_global_signal(
            bold, structures=global_signal_structures, axis=axis
        )

    if scale_confounds and confounds is not None and not confounds.empty:
        confounds = standardize_columns(confounds)
        
    frame_times = build_frame_times(n_scans, tr, start_time)

    design_matrix, three_column = create_design_matrix(
        events, confounds, config["regressors"], n_scans, frame_times
    )
    design_matrix, dropped = drop_zero_variance(design_matrix)
    contrasts, skipped = filter_contrasts(config.get("contrasts") or {}, dropped)

    result = {
        "design_matrix": design_matrix,
        "contrasts": contrasts,
        "dropped_columns": dropped,
        "skipped_contrasts": skipped,
        "three_column": three_column,
        "n_scans": n_scans,
        "tr": tr,
        "start_time": start_time,
        "n_dummy": n_dummy,
        "events": events,
        "frame_times": frame_times,
        "files": files,
    }
    if return_bold:
        result["bold"] = bold
    return result
