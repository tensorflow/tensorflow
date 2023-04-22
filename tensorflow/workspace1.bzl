"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("//third_party/android:android_configure.bzl", "android_configure")
load("@tf_toolchains//toolchains:archives.bzl", "bazel_toolchains_archive")
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")
load("@rules_cuda//cuda:dependencies.bzl", "rules_cuda_dependencies")

def workspace():
    native.register_toolchains("@local_config_python//:py_toolchain")
    rules_cuda_dependencies()

    closure_repositories()
    bazel_toolchains_archive()

    android_configure(name = "local_config_android")

    grpc_deps()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_workspace1 = workspace
