"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("//third_party/android:android_configure.bzl", "android_configure")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")
load("@rules_cuda//cuda:dependencies.bzl", "rules_cuda_dependencies")

def workspace():
    native.register_toolchains("@local_config_python//:py_toolchain")
    rules_cuda_dependencies()

    closure_repositories()

    http_archive(
        name = "bazel_toolchains",
        sha256 = "77c2c3c562907a1114afde7b358bf3d5cc23dc61b3f2fd619bf167af0c9582a3",
        strip_prefix = "bazel-toolchains-dfc67056200b674accd08d8f9a21e328098c07e2",
        urls = [
            "http://mirror.tensorflow.org/github.com/bazelbuild/bazel-toolchains/archive/dfc67056200b674accd08d8f9a21e328098c07e2.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/archive/dfc67056200b674accd08d8f9a21e328098c07e2.tar.gz",
        ],
    )

    android_configure(name = "local_config_android")

    grpc_deps()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_workspace1 = workspace
