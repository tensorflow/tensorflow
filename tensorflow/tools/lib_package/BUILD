# Packaging the TensorFlow C API into a small, standalone archive for use with
# language bindings and installations without Python.
#
# TODO(ashankar): Something similar for the JNI library for Java?
# TODO(ashankar): Something similar for the C++ API (caveat: ABI compatibility)

package(default_visibility = ["//visibility:private"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "libtensorflow",
    extension = "tar.gz",
    deps = [
        ":cheaders",
        ":clib",
        ":clicenses",
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

pkg_tar(
    name = "clicenses",
    files = [":include/tensorflow/c/LICENSE"],
    package_dir = "include/tensorflow/c",
)

genrule(
    name = "clicenses_generate",
    srcs = [
        "//third_party/hadoop:LICENSE.txt",
        "//third_party/eigen3:LICENSE",
        "@boringssl//:LICENSE",
        "@com_googlesource_code_re2//:LICENSE",
        "@curl//:COPYING",
        "@eigen_archive//:COPYING.MPL2",
        "@farmhash_archive//:COPYING",
        "@gemmlowp//:LICENSE",
        "@gif_archive//:COPYING",
        "@grpc//:LICENSE",
        "@highwayhash//:LICENSE",
        "@jemalloc//:COPYING",
        "@jpeg//:LICENSE.md",
        "@libxsmm_archive//:LICENSE",
        "@local_config_sycl//sycl:LICENSE.text",
        "@nanopb_git//:LICENSE.txt",
        "@png_archive//:LICENSE",
        "@protobuf//:LICENSE",
        "@zlib_archive//:zlib.h",
    ],
    outs = ["include/tensorflow/c/LICENSE"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)

sh_test(
    name = "libtensorflow_test",
    size = "small",
    srcs = ["libtensorflow_test.sh"],
    data = [
        "libtensorflow_test.c",
        ":libtensorflow.tar.gz",
    ],
)
