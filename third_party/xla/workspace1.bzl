"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")
load("@local_tsl//:workspace1.bzl", "tsl_workspace1")
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

# buildifier: disable=unnamed-macro
def workspace():
    """Loads a set of TensorFlow dependencies in a WORKSPACE file."""
    tsl_workspace1()

    native.register_toolchains("@local_config_python//:py_toolchain")
    rules_pkg_dependencies()

    closure_repositories()

    http_archive(
        name = "bazel_toolchains",
        sha256 = "294cdd859e57fcaf101d4301978c408c88683fbc46fbc1a3829da92afbea55fb",
        strip_prefix = "bazel-toolchains-8c717f8258cd5f6c7a45b97d974292755852b658",
        urls = [
            "http://mirror.tensorflow.org/github.com/bazelbuild/bazel-toolchains/archive/8c717f8258cd5f6c7a45b97d974292755852b658.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/archive/8c717f8258cd5f6c7a45b97d974292755852b658.tar.gz",
        ],
    )

    grpc_deps()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
xla_workspace1 = workspace
