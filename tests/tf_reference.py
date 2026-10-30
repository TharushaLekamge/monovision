"""Load the ORIGINAL AU-GAN TensorFlow code (third_party/AU-GAN) under TF2's v1-compat mode.

The upstream code targets TF 1.14 (tf.contrib, scipy.misc). We copy it to a temp dir at test
time and apply minimal text patches so the graph can be built with modern TensorFlow. No upstream
code is vendored into this repository (AU-GAN ships without a licence).

Patches applied (none change the maths):
  * `import tensorflow as tf`           -> `import tensorflow.compat.v1 as tf`
  * `tensorflow.contrib.slim`           -> `tf_slim`
  * tf.contrib xavier initializer       -> removed (only used by an unused helper)
  * scipy.misc / matplotlib imports     -> stubbed (only used for image I/O, not the graph)
"""
import importlib
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, "..", "third_party", "AU-GAN")

PATCHES = {
    "ops.py": [
        ("import tensorflow as tf\n", "import tensorflow.compat.v1 as tf\n"),
        ("import tensorflow.contrib.slim as slim", "import tf_slim as slim"),
        ("import tensorflow.contrib as tf_contrib\n", ""),
        ("weight_init = tf_contrib.layers.xavier_initializer()", "weight_init = None"),
    ],
    "loss_utils.py": [("import tensorflow as tf\n", "import tensorflow.compat.v1 as tf\n")],
    "utils.py": [("import scipy.misc\n", "")],
    "models.py": [],
    "AUGAN.py": [("import matplotlib.pyplot as plt\n", "")],
}


FLOAT64_PATCHES = {
    "ops.py": [('initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))',
                'initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32), dtype=tf.float64)'),
               ('offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))',
                'offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0), dtype=tf.float64)')],
    "models.py": [("tf.image.resize_bicubic(image, [get_shape(image)[1] // (2 ** i), get_shape(image)[2] // (2 ** i)])",
                   "image[:, ::2 ** i, ::2 ** i, :]")],
}


def load(src=None, disable_noise=True, float64=False):
    """Return a namespace with the patched upstream modules: models, loss_utils, ops, AUGAN.

    float64=True rebuilds the upstream graph in double precision (test-only), so gradient
    parity can be checked without float32 round-off. TF's resize_bicubic only computes in
    float32, so in this mode it is replaced by strided subsampling, which is mathematically
    identical for AU-GAN's exact 2x downscale (forward and adjoint are checked separately in
    test_legacy_bicubic / test_bicubic_2x_is_subsampling).
    """
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    src = os.path.abspath(src or os.environ.get("AUGAN_SRC", DEFAULT_SRC))
    if not os.path.isfile(os.path.join(src, "models.py")):
        raise FileNotFoundError(f"AU-GAN source not found at {src}. Run scripts/00_setup.sh first.")
    tmp = tempfile.mkdtemp(prefix="augan_ref_")
    for fname, patches in PATCHES.items():
        with open(os.path.join(src, fname)) as f:
            code = f.read()
        for old, new in patches + (FLOAT64_PATCHES.get(fname, []) if float64 else []):
            if old not in code:
                raise RuntimeError(f"Upstream {fname} changed; patch target not found: {old!r}")
            code = code.replace(old, new)
        if fname != "ops.py":
            code = code.replace("import tensorflow as tf\n", "import tensorflow.compat.v1 as tf\n")
        if float64:
            code = code.replace("tf.float32", "tf.float64")
        with open(os.path.join(tmp, fname), "w") as f:
            f.write(code)
    sys.path.insert(0, tmp)
    try:
        for m in ["ops", "utils", "loss_utils", "models", "AUGAN"]:
            sys.modules.pop(m, None)
        mods = {m: importlib.import_module(m) for m in ["ops", "utils", "loss_utils", "models", "AUGAN"]}
    finally:
        sys.path.remove(tmp)
        shutil.rmtree(tmp, ignore_errors=True)
    if disable_noise:
        # The reconstruction branch adds N(0, 0.02) noise; disable it so outputs are deterministic.
        mods["models"].gaussian_noise_layer = lambda x, std: x
    return type("Ref", (), mods)
