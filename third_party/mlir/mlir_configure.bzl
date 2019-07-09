"""Repository rule to setup the external MLIR repository."""

_MLIR_REV = "0682b094816eb2b6afbe6630a20bdc15dfbb4649"
_MLIR_SHA256 = "7bb875f2df11bf49ac66a5b0e75985e8a588c81d176f6081af0b6e834c7b8505"

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
