"""Repository rule for MLIR autoconfiguration."""

def _mlir_configure_impl(repository_ctx):
    repository_ctx.file("WORKSPACE", "")
    label = Label("@org_tensorflow//third_party/mlir:mlir_configure.bzl")
    for entry in repository_ctx.path(label).dirname.readdir():
        repository_ctx.symlink(entry, entry.basename)

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
