"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")

# buildifier: disable=function-docstring
# buildifier: disable=unnamed-macro
def workspace():
    # Declares @tsl
    local_repository(
        name = "tsl",
        path = paths.join(Label("//:__pkg__").workspace_root, "third_party/tsl"),
    )

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
xla_workspace4 = workspace
