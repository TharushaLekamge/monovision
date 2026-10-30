---
tags: [legal, third-party]
updated: 2026-10-21
---
# Third-Party and Licensing

Source of truth: `THIRD_PARTY.md`.

| Component | Use | Licence | Pinned |
|---|---|---|---|
| AU-GAN | submodule, parity reference | **none** (all rights reserved) | `f65c7fa` (2026-10-10) |
| SimpleRecon | submodule, reconstruction | Niantic, **non-commercial** | `113b825` (2026-10-10) |
| ios_logger | capture app, built separately | see its repo | n/a |
| BDD100K | GAN training data | dataset terms | n/a |
| ScanNet | 3D evaluation | access request + terms | n/a |

SimpleRecon is used through `patches/simplerecon/` applied to a gitignored copy (`build/simplerecon`), not by editing the submodule.

## Open issues
- No project licence chosen yet (`LICENSE` exists; confirm it).
- `monovision/enhance` is a port written here with no AU-GAN source copied, but whether a port is a derivative work of an unlicensed repo is a legal question. Resolve before publishing or sharing.
- SimpleRecon's non-commercial licence constrains any downstream use.
