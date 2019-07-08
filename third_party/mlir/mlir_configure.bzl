"""Repository rule to setup the external MLIR repository."""

_MLIR_REV = "705b787de70da8e3fb86913d31510fe72a522633"
_MLIR_SHA256 = "e6cc6a021c367a33c22c98355b4fc07c87ebf115af3fb5ed9addec4e74988002"

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
