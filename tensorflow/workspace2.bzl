"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("//tensorflow:version_check.bzl", "check_bazel_version_at_least")
load("//tensorflow:workspace.bzl", "tf_repositories")

def workspace():
    # Check the bazel version before executing any repository rules, in case
    # those rules rely on the version we require here.
    check_bazel_version_at_least("1.0.0")

    # Load tf_repositories() before loading dependencies for other repository so
    # that dependencies like com_google_protobuf won't be overridden.
    tf_repositories()
