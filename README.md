# network-fmri

Tools for analysing the Network grant fMRI data on Sherlock.

Right now the package does one thing: **build a first-level design matrix for
one run, and load the corresponding surface data.** It handles the fiddly parts
— finding the right files, reading the TR and slice-timing reference from the
sidecar, trimming the warmup volumes, shifting the event onsets to match, and
pulling the confounds you ask for. What you do with the design matrix after
that is up to you.

Currently fsLR (den-91k) only. The design-matrix construction itself is
space-agnostic; adding a volumetric path would mean changing the file glob and
the time axis in two functions.

---

## Setup

### 1. Make your uv cache directory

uv caches downloaded packages. Point it somewhere on OAK rather than your home
directory, which is small. The lab keeps these together:

```bash
mkdir -p /oak/stanford/groups/russpold/uv_cache_pool/uv_cache_YOURSUNETID
```

Use your own SUNet ID. Sharing a pool directory means we can see each other's
caches, but each person writes to their own.

### 2. Edit `setup_uv_sherlock.sh`

Open it and replace the cache path with the one you just made. Here is what
each line does:

```bash
umask 0006          # new files are group-readable and group-writable, so
                    # others in the lab can read your work (LAB REQURIEMENT)

module load fsl     # only needed if you call FSL commands; harmless otherwise
module load python/3.12.1   # the Python uv will build the environment around
module load gcc/12.4.0      # compiler, needed by a few packages
module load uv              # uv itself, provided by Sherlock

export UV_PYTHON=$(which python3)   # tells uv to use the module's Python
                                    # rather than hunting for its own

export UV_CACHE_DIR="...YOURSUNETID"   # where downloaded packages are stored
```

### 3. Source it

```bash
source setup_uv_sherlock.sh
```

**Every time you open a new terminal.** Without it `uv` is not on your path and
the wrong Python is active. If something mysteriously stops working, this is
the first thing to check.

### 4. Build the environment

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock` and creates a `.venv` with exactly
the package versions everyone else has. Takes a minute or two the first time.

Run things with `uv run`, which uses that environment:

```bash
uv run python my_script.py
```

When setting up jobs to run on slurm, use this syntax, which locates your uv project directory so uv run works:
```
uv --directory "$PROJECT_ROOT" run "${SCRIPT_DIR}/my_script.py" <args if needed>
```

---

## What's in here

```
network-fmri/
├── pyproject.toml          # dependencies and package metadata
├── uv.lock                 # exact versions — this is what makes environments match
├── setup_uv_sherlock.sh    # module loads and environment variables (edit this)
├── src/
│   └── network_fmri/       # the importable package
│       ├── __init__.py     # marks the directory as a package
│       ├── model_setup.py  # everything: file finding, loading, design matrices
│       └── tasks/          # one YAML per task, defining its regressors
└── analyses/
    └── tutorial/
        └── tutorial.ipynb  # start here
```

### How importing works

If you have not used a Python package before: the code in `src/network_fmri/`
is *installed* into the environment by `uv sync`, so you can import it from
anywhere without worrying about file paths.

```python
from network_fmri import model_setup as dm

res = dm.build_run_design_matrix(...)
```

That works whether you are in `analyses/tutorial/`, in a script, or in a
notebook somewhere else entirely. You never need `sys.path` or a relative
import.

The task YAMLs are bundled with the package the same way, so
`dm.resolve_task_yaml("flanker")` finds them regardless of where you are
working. You do not have to know where they live on disk.

### The task YAMLs

Each task's model is defined entirely in its YAML file. Every regressor has the
same three fields:

```yaml
congruent:
  amplitude: 1                # a number, or an events column for parametric modulation
  duration: duration          # an events column, or a number for a fixed epoch
  subset: >-                  # a pandas query selecting rows from the events file
    trial_id == 'test_trial' and trial_type == 'congruent'
    and key_press == correct_response and response_time >= 0.2
```

Every name in a YAML is a real column in the events.tsv, so you can open the
two side by side and see exactly what the model is. Adding a task means writing
one YAML — no Python changes.

Only the eight single-task paradigms are here. The dual-task configs were set
aside because there is little or no data for them in the discovery sample and
their definitions have never been checked against real events.

---

## Start here

Run `analyses/tutorial/tutorial.ipynb`. On sherlock I'm assuming you're using the code server (VSCode) and you'll see a "select kernel" button in the upper right corner of the window after you open the notebook.  Click and select the local `.venv` that was created when you ran `uv sync` so the code works! It walks through building a design
matrix, choosing confounds, plotting the result, and fitting a model to get
residuals.

**Then come ask me at least two questions about how the code works.** Stupid
questions are my favorite kind — if something is unclear it is almost always
the code or the docs that need fixing, not you.

---

## Things you need to know about the data

**The OAK derivatives are untrimmed.** The first 7 volumes of each run are
warmup (non-steady-state) volumes, and the event onsets have not been adjusted
for them. This package removes those volumes, shifts the onsets by
−7 × TR = −10.43 s, and drops the matching rows from the confounds. That is
what `data_is_trimmed=False` means.

It is a required argument with no default, because **nothing in the files
records whether this has already been done.** A wrong value does not raise — it
silently misaligns your design matrix by 10.43 s. If the canonical data are
ever replaced with pre-trimmed versions, this needs testing before anyone
flips the flag; it has never been used with `True`.

Two manual checks if you want to confirm what state your data are in:

1. The number of volumes in the dtseries, against the expected raw length.
2. The first `test_trial` onset in the events file. Raw, it is around 11.9 s.
   Already shifted, it would be around 1.5 s.

**TR comes from the fMRIPrep sidecar, never from a config.** It is cross-checked
against the task YAML and a disagreement raises. This matters: one run in the
discovery sample (`s19 ses-04 shapeMatching`) was acquired at TR 2.264 rather
than 1.49, and the previous pipeline processed it at the wrong TR.

**The `cosine*` confounds are the high-pass filter.** They are fMRIPrep's
discrete cosine basis. If you include them, do not also pass a `high_pass` or
`drift_model` to the GLM, or your data get filtered twice. If you exclude them,
you need one of those instead.

---

## Adding your own code

**One-off analyses go in `analyses/`,** one directory per analysis step:

```
analyses/
├── tutorial/
├── 01_build_designs/
├── 02_fit_residuals/
└── 03_connectivity/
```

One step per directory means each has its own notebook or script plus whatever
it produces, and someone reading the repo can see the order things happened in.
Resist putting three unrelated things in one directory — when you come back in
six months, "which notebook made this figure" should be obvious.

**Anything reusable goes in `src/network_fmri/`** as a new module. The test is
whether you would want to call it from more than one analysis. A function that
fits a GLM and returns residuals belongs in the package; a notebook that fits
one particular subject's data does not.

New modules sit beside `model_setup.py`:

```
src/network_fmri/
├── model_setup.py
└── fit.py          # for example
```

and import from each other normally:

```python
from network_fmri.model_setup import build_run_design_matrix
```

After adding a module you do not need to reinstall — `uv sync` installs the
package in editable mode, so your edits take effect immediately.

---

## A note on the pinned versions

`pyproject.toml` pins several packages below their latest release. This is not
arbitrary.

Sherlock runs CentOS 7, whose glibc is 2.17 — the oldest version the Python
wheel system still supports. Packages are steadily dropping that support, and
when they do there is no pre-built wheel our system can use, so pip and uv fall
back to compiling from source. Those builds fail, for reasons ranging from
missing system libraries to an assembler too old to understand the instructions
numpy's source uses.

So each compiled package is pinned below the last version that still ships a
`manylinux2014` wheel. Pure-Python packages (nibabel, nilearn, pyyaml, seaborn)
are unaffected and unpinned.

Stanford Research Computing has confirmed there is no timeline for an OS
upgrade, so expect this to need occasional maintenance: when you want a newer
version of something, check whether it still publishes a `manylinux2014` wheel
for cp312 on PyPI before raising the ceiling.