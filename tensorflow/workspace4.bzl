"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_vendored")

def workspace():
    tf_vendored(name = "local_xla", relpath = "third_party/xla")
    tf_vendored(name = "local_tsl", relpath = "third_party/xla/third_party/tsl")

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_workspace4 = workspace
