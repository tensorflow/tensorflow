# Packaging for TensorFlow artifacts other than the Python API (pip whl).
# This includes the C API, Java API, and protocol buffer files.

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_xla//xla/tsl/mkl:build_defs.bzl", "if_enable_mkl", "if_mkl")
load("@rules_pkg//:pkg.bzl", "pkg_tar", "pkg_zip")
load("//tensorflow:tensorflow.bzl", "VERSION", "VERSION_MAJOR", "if_macos")
load("//tensorflow/core/platform:build_config_root.bzl", "tf_additional_license_deps")

package(default_visibility = ["//visibility:private"])

pkg_zip(
    name = "libtensorflow_proto",
    srcs = ["//tensorflow/core:protos_all_proto_srcs"],
    strip_prefix = "/external/local_tsl",
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
    srcs = [
        ":THIRD_PARTY_TF_JNI_LICENSES",
        "//:LICENSE",
        "//tensorflow/java:libtensorflow_jni",
    ],
    extension = "tar.gz",
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
    srcs = ["//tensorflow:tensorflow_framework"],
    symlinks = if_macos(
        {
            "libtensorflow_framework.dylib": "libtensorflow_framework.%s.dylib" % VERSION_MAJOR,
            "libtensorflow_framework.%s.dylib" % VERSION_MAJOR: "libtensorflow_framework.%s.dylib" % VERSION,
        },
        {
            "libtensorflow_framework.so": "libtensorflow_framework.so.%s" % VERSION_MAJOR,
            "libtensorflow_framework.so.%s" % VERSION_MAJOR: "libtensorflow_framework.so.%s" % VERSION,
        },
    ),
    tags = ["manual"],
)

pkg_tar(
    name = "cheaders",
    srcs = [
        "//tensorflow/c:headers",
    ],
    package_dir = "include/",
    strip_prefix = "/external/local_xla",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
)

pkg_tar(
    name = "eager_cheaders",
    srcs = [
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
    srcs = ["//tensorflow"],
    package_dir = "lib",
    symlinks = if_macos(
        {
            "lib/libtensorflow.dylib": "libtensorflow.%s.dylib" % VERSION_MAJOR,
            "lib/libtensorflow.%s.dylib" % VERSION_MAJOR: "libtensorflow.%s.dylib" % VERSION,
        },
        {
            "lib/libtensorflow.so": "libtensorflow.so.%s" % VERSION_MAJOR,
            "lib/libtensorflow.so.%s" % VERSION_MAJOR: "libtensorflow.so.%s" % VERSION,
        },
    ),
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
    srcs = [
        ":THIRD_PARTY_TF_C_LICENSES",
        "//:LICENSE",
    ],
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
        "//third_party/icu/data:LICENSE",
        "@curl//:COPYING",
        "@ducc//:LICENSE",
        "@gemmlowp//:LICENSE",
        "@libjpeg_turbo//:LICENSE.md",
        "@llvm-project//llvm:LICENSE.TXT",
        "@llvm-project//mlir:LICENSE.TXT",
        "@local_config_tensorrt//:LICENSE",
        "@org_brotli//:LICENSE",
        "@png//:LICENSE",
        "@snappy//:COPYING",
        "@stablehlo//:LICENSE",
        "@tf_runtime//:LICENSE",
        "@local_tsl//:LICENSE",
        "@local_xla//:LICENSE",
    ] + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_gcp_support": [],
        "//conditions:default": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
    }) + select({
        "@local_xla//xla/tsl:with_numa_support": [
            "@hwloc//:COPYING",
        ],
        "//conditions:default": [],
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
        "@local_config_nccl//:LICENSE",
    ]) + if_mkl([
        "//third_party/mkl_dnn:LICENSE",
    ]) + if_enable_mkl(["@local_xla//xla/tsl/mkl:LICENSE"]) + tf_additional_license_deps(),
    outs = ["THIRD_PARTY_TF_C_LICENSES"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)

genrule(
    name = "jnilicenses_generate",
    srcs = [
        "//third_party/icu/data:LICENSE",
        "@curl//:COPYING",
        "@ducc//:LICENSE",
        "@gemmlowp//:LICENSE",
        "@libjpeg_turbo//:LICENSE.md",
        "@llvm-project//llvm:LICENSE.TXT",
        "@llvm-project//mlir:LICENSE.TXT",
        "@local_config_tensorrt//:LICENSE",
        "@org_brotli//:LICENSE",
        "@png//:LICENSE",
        "@snappy//:COPYING",
        "@stablehlo//:LICENSE",
        "@tf_runtime//:LICENSE",
        "@local_tsl//:LICENSE",
        "@local_xla//:LICENSE",
    ] + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_gcp_support": [],
        "//conditions:default": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
    }) + select({
        "@local_xla//xla/tsl:with_numa_support": [
            "@hwloc//:COPYING",
        ],
        "//conditions:default": [],
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
        "@local_config_nccl//:LICENSE",
    ]) + if_mkl([
        "//third_party/mkl_dnn:LICENSE",
    ]) + if_enable_mkl(["@local_xla//xla/tsl/mkl:LICENSE"]) + tf_additional_license_deps(),
    outs = ["THIRD_PARTY_TF_JNI_LICENSES"],
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
