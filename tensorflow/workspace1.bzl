"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("//third_party/android:android_configure.bzl", "android_configure")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")
load("@rules_cuda//cuda:dependencies.bzl", "rules_cuda_dependencies")
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")

def workspace():
    native.register_toolchains("@local_config_python//:py_toolchain")
    rules_cuda_dependencies()
    rules_pkg_dependencies()

    closure_repositories()

    http_archive(
        name = "bazel_toolchains",
        sha256 = "540cc8fec2bf8ab64d16fb9a7018f25738a4a03434057ea01b5d34add446ffb1",
        strip_prefix = "bazel-toolchains-ea243d43269df23de03a797cff2347e1fc3d02bb",
        urls = [
            "http://mirror.tensorflow.org/github.com/bazelbuild/bazel-toolchains/archive/ea243d43269df23de03a797cff2347e1fc3d02bb.tar.gz",
            "https://github.com/bazelbuild/bazel-toolchains/archive/ea243d43269df23de03a797cff2347e1fc3d02bb.tar.gz",
        ],
    )

    android_configure(name = "local_config_android")

    grpc_deps()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_workspace1 = workspace
