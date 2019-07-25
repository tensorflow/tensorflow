"""Repository rule to setup the external MLIR repository."""

_MLIR_REV = "83ff81bfd9d382852d0302ab2a234feb2e938fc7"
_MLIR_SHA256 = "26979670616980014a823f88c1a057c28080763d9cb189fa67172a92c085d349"

def _mlir_autoconf_impl(repository_ctx):
    """Implementation of the mlir_configure repository rule."""
    repository_ctx.download_and_extract(
        [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/tensorflow/mlir/archive/{}.zip".format(_MLIR_REV),
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
