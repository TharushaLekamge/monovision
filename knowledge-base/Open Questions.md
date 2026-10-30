---
tags: [questions]
updated: 2026-10-21
---
# Open Questions

- Where exactly does the paper place upsample+conv and demodulation? (see [[Decisions and Assumptions]])
- Does the `_00` suffix on ScanNet scan IDs match the paper's test set?
- Is a port of an unlicensed repo safe to publish? ([[Third-Party and Licensing]])
- Why does TF + PyTorch in one process crash natively? (worked around with separate processes)
- Which licence for MonoVision, given SimpleRecon's non-commercial terms?
- Should `knowledge-base/` be committed to the repo?
- ~~Can SimpleRecon run on an RTX 5060?~~ Yes, via the port ([[SimpleRecon Port]]).
- Does the ported SimpleRecon match the original numerically? No reference available here.
- Publish the SimpleRecon patches as a fork? (Niantic non-commercial licence)
- What batch size / resolution to train with on 8 GB VRAM?
