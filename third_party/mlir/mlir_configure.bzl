"""Repository rule for MLIR autoconfiguration."""

def _mlir_configure_impl(repository_ctx):
    repository_ctx.file("WORKSPACE", "")
    label = Label("@org_tensorflow//third_party/mlir:mlir_configure.bzl")
    src_dir = repository_ctx.path(label).dirname
    repository_ctx.execute(["cp", "-rLf", "%s/." % src_dir, "."])

mlir_configure = repository_rule(
    implementation = _mlir_configure_impl,
)
"""Detects and configures the MLIR configuration.

Add the following to your WORKSPACE FILE:

```python
mlir_configure(name = "local_config_mlir")
```

Args:
  name: A unique name for this workspace rule.
"""
