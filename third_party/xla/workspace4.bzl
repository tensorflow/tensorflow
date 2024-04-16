"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_vendored")

# buildifier: disable=function-docstring
# buildifier: disable=unnamed-macro
def workspace(tsl_name = "tsl"):
    # Declares @local_tsl
    tf_vendored(name = tsl_name, relpath = "third_party/tsl")

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
xla_workspace4 = workspace
