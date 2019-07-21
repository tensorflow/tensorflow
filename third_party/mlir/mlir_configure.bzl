"""Repository rule to setup the external MLIR repository."""

_MLIR_REV = "5879f6e6432ae0bb927a02145a84ad1d945491ec"
_MLIR_SHA256 = "b6da7089f3631cef8c80548aac94febf2d39782a3b26c5ec564f19e7d786e343"

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
