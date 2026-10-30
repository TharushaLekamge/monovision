# Third-party code and data

| Component | Source | Used as | Licence | Pinned commit |
|---|---|---|---|---|
| AU-GAN | https://github.com/jgkwak95/AU-GAN | submodule `third_party/AU-GAN`. Reference for the port; loaded by the TF parity tests | **none in the repo** (all rights reserved by default) | `f65c7fa8863cfb980c5af57e73be3f9ce240cb87` (2026-10-10) |
| SimpleRecon | https://github.com/nianticlabs/simplerecon | submodule `third_party/simplerecon`. 3D reconstruction stage | Niantic licence, **non-commercial only** | `113b825084cc62166c6f5584d526650928127317` (2026-10-10) |
| ios_logger | https://github.com/Varvrar/ios_logger | capture app, built separately with Xcode | see its repo | — |
| BDD100K | https://bdd-data.berkeley.edu/ | GAN training data | dataset terms | — |
| ScanNet | http://www.scan-net.org/ | 3D evaluation | ScanNet terms of use (access request) | — |

`monovision/enhance` is a PyTorch port written for this repo: it reproduces AU-GAN's architecture and losses, but no AU-GAN source files are copied in. AU-GAN has no licence, and whether a port counts as a derivative work is a legal question. Check this before publishing or sharing the repo.

Commits are the upstream HEADs when this repo was set up (2026-10-10). To pin them:
`git -C third_party/<name> checkout <commit>`, then commit the submodule pointer.
