# Packaging for TensorFlow artifacts other than the Python API (pip whl).
# This includes the C API, Java API, and protocol buffer files.

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("@local_config_syslibs//:build_defs.bzl", "if_not_system_lib")
load("//tensorflow:tensorflow.bzl", "VERSION", "VERSION_MAJOR", "if_macos")
load("//tensorflow/core/platform:build_config_root.bzl", "tf_additional_license_deps")
load("//third_party/mkl:build_defs.bzl", "if_enable_mkl", "if_mkl")

package(default_visibility = ["//visibility:private"])

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
    strip_prefix = "/",
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
        "//third_party/eigen3:LICENSE",
        "//third_party/fft2d:LICENSE",
        "//third_party/icu/data:LICENSE",
        "@boringssl//:LICENSE",
        "@com_google_protobuf//:LICENSE",
        "@com_googlesource_code_re2//:LICENSE",
        "@curl//:COPYING",
        "@double_conversion//:LICENSE",
        "@eigen_archive//:COPYING.MPL2",
        "@farmhash_archive//:COPYING",
        "@fft2d//:readme2d.txt",
        "@gemmlowp//:LICENSE",
        "@gif//:COPYING",
        "@highwayhash//:LICENSE",
        "@icu//:icu4c/LICENSE",
        "@libjpeg_turbo//:LICENSE.md",
        "@llvm-project//llvm:LICENSE.TXT",
        "@llvm-project//mlir:LICENSE.TXT",
        "@lmdb//:LICENSE",
        "@local_config_tensorrt//:LICENSE",
        "@nasm//:LICENSE",
        "@nsync//:LICENSE",
        "@png//:LICENSE",
        "@snappy//:COPYING",
        "@zlib//:zlib.h",
    ] + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//conditions:default": [],
    }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_gcp_support": [],
        "//conditions:default": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
    }) + select({
        "//tensorflow/core/kernels:xsmm": [
            "@libxsmm_archive//:LICENSE.md",
        ],
        "//conditions:default": [],
    }) + select({
        "//tensorflow:with_numa_support": [
            "@hwloc//:COPYING",
        ],
        "//conditions:default": [],
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
        "@local_config_nccl//:LICENSE",
    ]) + if_mkl([
        "//third_party/mkl_dnn:LICENSE",
    ]) + if_enable_mkl(["//third_party/mkl:LICENSE"]) + if_not_system_lib(
        "com_github_grpc_grpc",
        [
            "@com_github_grpc_grpc//:LICENSE",
            "@com_github_grpc_grpc//third_party/address_sorting:LICENSE",
        ],
    ) + tf_additional_license_deps(),
    outs = ["THIRD_PARTY_TF_C_LICENSES"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)

genrule(
    name = "jnilicenses_generate",
    srcs = [
        "//third_party/eigen3:LICENSE",
        "//third_party/fft2d:LICENSE",
        "//third_party/icu/data:LICENSE",
        "@boringssl//:LICENSE",
        "@com_google_protobuf//:LICENSE",
        "@com_googlesource_code_re2//:LICENSE",
        "@curl//:COPYING",
        "@double_conversion//:LICENSE",
        "@eigen_archive//:COPYING.MPL2",
        "@farmhash_archive//:COPYING",
        "@fft2d//:readme2d.txt",
        "@gemmlowp//:LICENSE",
        "@gif//:COPYING",
        "@com_github_grpc_grpc//:LICENSE",
        "@com_github_grpc_grpc//third_party/address_sorting:LICENSE",
        "@highwayhash//:LICENSE",
        "@icu//:icu4j/main/shared/licenses/LICENSE",
        "@libjpeg_turbo//:LICENSE.md",
        "@llvm-project//llvm:LICENSE.TXT",
        "@llvm-project//mlir:LICENSE.TXT",
        "@lmdb//:LICENSE",
        "@local_config_tensorrt//:LICENSE",
        "@nasm//:LICENSE",
        "@nsync//:LICENSE",
        "@png//:LICENSE",
        "@snappy//:COPYING",
        "@zlib//:zlib.h",
    ] + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//conditions:default": [],
    }) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_gcp_support": [],
        "//conditions:default": [
            "@com_github_googlecloudplatform_google_cloud_cpp//:LICENSE",
        ],
    }) + select({
        "//tensorflow/core/kernels:xsmm": [
            "@libxsmm_archive//:LICENSE.md",
        ],
        "//conditions:default": [],
    }) + select({
        "//tensorflow:with_numa_support": [
            "@hwloc//:COPYING",
        ],
        "//conditions:default": [],
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
        "@local_config_nccl//:LICENSE",
    ]) + if_mkl([
        "//third_party/mkl_dnn:LICENSE",
    ]) + if_enable_mkl(["//third_party/mkl:LICENSE"]) + tf_additional_license_deps(),
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
