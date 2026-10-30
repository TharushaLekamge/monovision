#!/usr/bin/env bash
# One-time repo setup: git init + upstream code as pinned submodules.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .git ] || git init
add() {  # url path
  if [ ! -e "$2/.git" ]; then git submodule add "$1" "$2"; fi
  echo "$2 @ $(git -C "$2" rev-parse HEAD)"
}
add https://github.com/jgkwak95/AU-GAN third_party/AU-GAN
add https://github.com/nianticlabs/simplerecon third_party/simplerecon
echo "Record the commit hashes above in THIRD_PARTY.md."
