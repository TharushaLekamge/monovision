---
tags: [home]
updated: 2026-10-21
---
# MonoVision Knowledge Base

Shared notes for the user and Claude. Open the `knowledge-base/` folder as an Obsidian vault. Read this page first; keep it current.

## Navigate
- [[Project Overview]]: what the repo is, the paper, the pipeline
- [[Status]]: phase-by-phase state
- [[Pipeline]]: stage-by-stage data flow
- [[AU-GAN Port]]: the PyTorch port and how it was verified
- [[Environment]]: two conda envs, what has been tested
- [[Third-Party and Licensing]]: submodules, licences, open legal questions
- [[Decisions and Assumptions]]: choices made where the paper was silent
- [[Open Questions]]
- [[SimpleRecon Port]]: SimpleRecon running on current Python/torch via a patch series
- [[Build and Run Results]]: first build/test/run on this machine (2026-10-21)
- [[Changelog]]: dated log of what changed

## TODO
Mirrors the phases in [[Status]]. Tick items here as they land, and add a line to [[Changelog]].

### Phase 1: Repo and environment
- [x] Submodules present and `monovision` env builds; `check_env.py` passes ([[Build and Run Results]])
- [x] SimpleRecon runs in the monovision env on this GPU ([[SimpleRecon Port]])
- [ ] Create the `simplerecon` env and run `python scripts/check_env.py --recon`
- [ ] Test on Python 3.9 / torch 1.10 (tested so far: 3.13 / torch 2.14 CPU, 3.11 / torch 2.11 CPU)
- [ ] Choose a LICENSE (see [[Third-Party and Licensing]]); the file `LICENSE` is tracked but check it is the intended one
- [ ] Commit the untracked work (`monovision/`, `docs/`, `tests/`, `configs/`, `scripts/`, ...); only the initial commit exists
- [ ] Decide whether to pin submodule commits (`git -C third_party/<name> checkout <commit>`)

### Phase 2: Data
- [ ] Download BDD100K and build the night/day split (`trainA/trainB/testA/testB`)
- [ ] Request ScanNet access; fetch the 16 test scans in `data_splits/scannet16.txt`
- [ ] Verify the `_00` suffix assumption on the 16 scan IDs
- [ ] Build `monovision/capture`: ios_logger → frames
- [ ] Build `monovision/data`: synthetic night generation for ScanNet

### Phase 3: SimpleRecon baselines
- [x] Port SimpleRecon inference to Python 3.11 / current torch; run on Niantic's sample scans
- [ ] Get a reference number to confirm the port matches the original (no pinned env available here)
- [ ] Run the port on an ios_logger scan (`datasets/arkit_dataset.py`, `data_scripts/ios_logger_preprocessing.py`), untested
- [x] Port SimpleRecon training (branch `simplerecon-training`, patch 0005); smoke-tested on one GPU ([[SimpleRecon Port]])
- [x] Address python-pr-review findings (tests, script, moviepy fix, loader, pins)
- [ ] Merge `simplerecon-training` into `main` once the main work is committed (worktree `../monovision-train`)
- [ ] Train/fine-tune on real data (ScanNet) and compare to the paper
- [ ] Review the 4 (5 on the branch) patches in `patches/simplerecon/` and decide whether to publish them as a fork
- [x] Download SimpleRecon weights (hero_model; `scripts/01_setup_simplerecon.sh --weights`)
- [ ] Run baseline on day ScanNet, record mesh metrics
- [ ] Run baseline on synthetic night, record the degradation

### Phase 4: GAN
- [ ] Pick training settings that fit 8 GB: configs use batch 8 at 256×256, which OOMs; pt-AUGAN fits at batch 1, original AU-GAN needs a smaller size
- [ ] Train `augan_bdd100k.yaml` on a GPU (paper: one V100, ~10 h/epoch)
- [ ] Train `pt_augan_bdd100k.yaml`
- [ ] Optional: convert the original TF checkpoint with `convert_tf.py` and compare
- [ ] Confirm pt-AUGAN assumptions against the paper (see [[Decisions and Assumptions]])

### Phase 5: Pipeline glue
- [ ] Implement `monovision run` in `monovision/recon` (enhance in `monovision` env → reconstruct in `simplerecon` env, exchanging image folders)
- [ ] Check frame filenames and camera poses stay aligned end to end

### Phase 6: Evaluation
- [ ] FID for the GAN output
- [ ] Mesh precision / recall / F-score
- [ ] Significance tests
- [ ] Compare against the paper's reported numbers

### KB housekeeping
- [ ] Open the vault in Obsidian and check links resolve
- [ ] Decide whether `knowledge-base/` is committed or ignored
