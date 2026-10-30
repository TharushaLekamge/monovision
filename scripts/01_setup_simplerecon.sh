#!/usr/bin/env bash
# Build the ported SimpleRecon: copy the pinned submodule to build/simplerecon (the submodule
# itself stays untouched) and apply patches/simplerecon/*.patch. build/simplerecon is
# disposable and recreated on every run. The checkpoint is kept in build/weights and linked in,
# so re-running never deletes it.
#   --weights   also fetch hero_model.ckpt (~200 MB, Niantic's Google Drive) and verify its
#               SHA-256. The hash was taken from the file downloaded on 2026-10-21 (trust on
#               first use; Niantic publishes none).
# SIMPLERECON_BUILD_DIR overrides the output directory (used by the tests).
set -euo pipefail
cd "$(dirname "$0")/.."
src=third_party/simplerecon
dst=${SIMPLERECON_BUILD_DIR:-build/simplerecon}
weights_store="$(dirname "$dst")/weights"
ckpt=hero_model.ckpt
ckpt_sha256=4752de233de1253934434141c2f713d7e0282326457386604477640d90028b74
ckpt_url="https://drive.usercontent.google.com/download?id=1hCuKZjEq-AghrYAmFxJs_4eeixIlP488&export=download&confirm=t"

[ -f "$src/test.py" ] || { echo "submodule missing: bash scripts/00_setup.sh"; exit 1; }
rm -rf "${dst:?}" && mkdir -p "$dst"
tar -C "$src" --exclude=.git -cf - . | tar -C "$dst" -xf -
for p in patches/simplerecon/*.patch; do
  patch -s -p1 --no-backup-if-mismatch -d "$dst" -i "$PWD/$p"   # not git apply: it skips paths inside a repo subdir
  echo "applied $(basename "$p")"
done

if [ "${1:-}" = "--weights" ] && [ ! -f "$weights_store/$ckpt" ]; then
  mkdir -p "$weights_store"
  part="$weights_store/$ckpt.part"
  curl -fL -o "$part" "$ckpt_url"
  if ! echo "$ckpt_sha256  $part" | sha256sum -c --status; then
    rm -f "${part:?}"
    echo "checksum mismatch for $ckpt (Drive may have returned an error page); removed"; exit 1
  fi
  mv "$part" "$weights_store/$ckpt"
fi
if [ -f "$weights_store/$ckpt" ]; then
  ln -sf "$(cd "$weights_store" && pwd)/$ckpt" "$dst/weights/$ckpt"
fi
echo "ready: $dst  (run from there: python test.py ...)"
