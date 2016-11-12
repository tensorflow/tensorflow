# Packaging the TensorFlow C API into a small, standalone archive for use with
# language bindings and installations without Python.
#
# TODO(ashankar): Also package the C++ API as a separate archive
# TODO(ashankar): Create pkg_deb rules

package(default_visibility = ["//visibility:private"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar", "pkg_deb")

pkg_tar(
    name = "libtensorflow",
    extension = "tar.gz",
    deps = [
        ":cheaders",
        ":clib",
    ],
)

pkg_tar(
    name = "cheaders",
    files = ["//tensorflow/c:headers"],
    package_dir = "include/tensorflow/c",
)

pkg_tar(
    name = "clib",
    files = ["//tensorflow:libtensorflow.so"],
    package_dir = "lib",
)
