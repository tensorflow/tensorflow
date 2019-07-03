"""Repository rule to setup the external MLIR repository."""

_MLIR_REV = "57c6e29a5638e9d2920550e3e6285001ee4015a0"
_MLIR_SHA256 = "c65805b04647b391e24db12251a2c7c2c382446b563604da1b2d05fa3854fa7c"

def _mlir_autoconf_impl(repository_ctx):
    """Implementation of the mlir_configure repository rule."""
    repository_ctx.download_and_extract(
        [
            "http://mirror.tensorflow.org/github.com/tensorflow/mlir/archive/{}.zip".format(_MLIR_REV),
            "https://github.com/tensorflow/mlir/archive/{}.zip".format(_MLIR_REV),
        ],
        sha256 = _MLIR_SHA256,
        stripPrefix = "mlir-{}".format(_MLIR_REV),
    )

    # Merge the checked-in BUILD files into the downloaded repo.
    for file in ["BUILD", "tblgen.bzl", "test/BUILD"]:
        repository_ctx.template(file, Label("//third_party/mlir:" + file))

mlir_configure = repository_rule(
    implementation = _mlir_autoconf_impl,
)
"""Configures the MLIR repository.

Add the following to your WORKSPACE FILE:

```python
mlir_configure(name = "local_config_mlir")
```

Args:
  name: A unique name for this workspace rule.
"""
