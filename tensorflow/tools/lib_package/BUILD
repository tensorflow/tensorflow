# Packaging for TensorFlow artifacts other than the Python API (pip whl).
# This includes the C API, Java API, and protocol buffer files.

package(default_visibility = ["//visibility:private"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("@local_config_syslibs//:build_defs.bzl", "if_not_system_lib")
load("//tensorflow:tensorflow.bzl", "tf_binary_additional_srcs")
load("//tensorflow:tensorflow.bzl", "if_cuda")
load("//third_party/mkl:build_defs.bzl", "if_mkl")

genrule(
    name = "libtensorflow_proto",
    srcs = ["//tensorflow/core:protos_all_proto_srcs"],
    outs = ["libtensorflow_proto.zip"],
    cmd = "zip $@ $(SRCS)",
)

pkg_tar(
    name = "libtensorflow",
    extension = "tar.gz",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [
        ":cheaders",
        ":clib",
        ":clicenses",
        ":eager_cheaders",
    ],
)

pkg_tar(
    name = "libtensorflow_jni",
    extension = "tar.gz",
    files = [
        "include/tensorflow/jni/LICENSE",
        "//tensorflow/java:libtensorflow_jni",
    ],
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [":common_deps"],
)

# Shared objects that all TensorFlow libraries depend on.
pkg_tar(
    name = "common_deps",
    files = tf_binary_additional_srcs(),
    tags = ["manual"],
)

pkg_tar(
    name = "cheaders",
    files = [
        "//tensorflow/c:headers",
    ],
    package_dir = "include/tensorflow/c",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
)

pkg_tar(
    name = "eager_cheaders",
    files = [
        "//tensorflow/c/eager:headers",
    ],
    package_dir = "include/tensorflow/c/eager",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
)

pkg_tar(
    name = "clib",
    files = ["//tensorflow:libtensorflow.so"],
    package_dir = "lib",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [":common_deps"],
)

pkg_tar(
    name = "clicenses",
    files = [":include/tensorflow/c/LICENSE"],
    package_dir = "include/tensorflow/c",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
)

genrule(
    name = "clicenses_generate",
    srcs = [
        "//third_party/hadoop:LICENSE.txt",
        "//third_party/eigen3:LICENSE",
        "//third_party/fft2d:LICENSE",
        "@boringssl//:LICENSE",
        "@com_googlesource_code_re2//:LICENSE",
        "@curl//:COPYING",
        "@double_conversion//:LICENSE",
        "@eigen_archive//:COPYING.MPL2",
        "@farmhash_archive//:COPYING",
        "@fft2d//:fft/readme.txt",
        "@gemmlowp//:LICENSE",
        "@gif_archive//:COPYING",
        "@highwayhash//:LICENSE",
        "@jpeg//:LICENSE.md",
        "@llvm//:LICENSE.TXT",
        "@lmdb//:LICENSE",
        "@local_config_sycl//sycl:LICENSE.text",
        "@nasm//:LICENSE",
        "@nsync//:LICENSE",
        "@png_archive//:LICENSE",
        "@protobuf_archive//:LICENSE",
        "@snappy//:COPYING",
        "@zlib_archive//:zlib.h",
    ] + select({
        "//tensorflow:with_aws_support": [
            "@aws//:LICENSE",
        ],
        "//conditions:default": [],
    }) + select({
        "//tensorflow:with_gcp_support": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
        "//conditions:default": [],
    }) + select({
        "//tensorflow:with_jemalloc_linux_x86_64": [
            "@jemalloc//:COPYING",
        ],
        "//tensorflow:with_jemalloc_linux_ppc64le": [
            "@jemalloc//:COPYING",
        ],
        "//conditions:default": [],
    }) + select({
        "//tensorflow/core/kernels:xsmm": [
            "@libxsmm_archive//:LICENSE.md",
        ],
        "//conditions:default": [],
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
    ]) + if_mkl([
        "//third_party/mkl:LICENSE",
        "//third_party/mkl_dnn:LICENSE",
    ]) + if_not_system_lib(
        "grpc",
        [
            "@grpc//:LICENSE",
            "@grpc//third_party/nanopb:LICENSE.txt",
            "@grpc//third_party/address_sorting:LICENSE",
        ],
    ),
    outs = ["include/tensorflow/c/LICENSE"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)

genrule(
    name = "jnilicenses_generate",
    srcs = [
        "//third_party/hadoop:LICENSE.txt",
        "//third_party/eigen3:LICENSE",
        "//third_party/fft2d:LICENSE",
        "@boringssl//:LICENSE",
        "@com_googlesource_code_re2//:LICENSE",
        "@curl//:COPYING",
        "@double_conversion//:LICENSE",
        "@eigen_archive//:COPYING.MPL2",
        "@farmhash_archive//:COPYING",
        "@fft2d//:fft/readme.txt",
        "@gemmlowp//:LICENSE",
        "@gif_archive//:COPYING",
        "@highwayhash//:LICENSE",
        "@jpeg//:LICENSE.md",
        "@llvm//:LICENSE.TXT",
        "@lmdb//:LICENSE",
        "@local_config_sycl//sycl:LICENSE.text",
        "@nasm//:LICENSE",
        "@nsync//:LICENSE",
        "@png_archive//:LICENSE",
        "@protobuf_archive//:LICENSE",
        "@snappy//:COPYING",
        "@zlib_archive//:zlib.h",
    ] + select({
        "//tensorflow:with_aws_support": [
            "@aws//:LICENSE",
        ],
        "//conditions:default": [],
    }) + select({
        "//tensorflow:with_gcp_support": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
        "//conditions:default": [],
    }) + select({
        "//tensorflow:with_jemalloc_linux_x86_64": [
            "@jemalloc//:COPYING",
        ],
        "//tensorflow:with_jemalloc_linux_ppc64le": [
            "@jemalloc//:COPYING",
        ],
        "//conditions:default": [],
    }) + select({
        "//tensorflow/core/kernels:xsmm": [
            "@libxsmm_archive//:LICENSE.md",
        ],
        "//conditions:default": [],
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
    ]) + if_mkl([
        "//third_party/mkl:LICENSE",
        "//third_party/mkl_dnn:LICENSE",
    ]),
    outs = ["include/tensorflow/jni/LICENSE"],
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
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    # Till then, this test is explicitly executed when building
    # the release by tensorflow/tools/ci_build/builds/libtensorflow.sh
    tags = ["manual"],
)

sh_test(
    name = "libtensorflow_java_test",
    size = "small",
    srcs = ["libtensorflow_java_test.sh"],
    data = [
        ":LibTensorFlowTest.java",
        ":libtensorflow_jni.tar.gz",
        "//tensorflow/java:libtensorflow.jar",
    ],
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    # Till then, this test is explicitly executed when building
    # the release by tensorflow/tools/ci_build/builds/libtensorflow.sh
    tags = ["manual"],
)
