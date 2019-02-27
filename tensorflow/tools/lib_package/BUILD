# Packaging for TensorFlow artifacts other than the Python API (pip whl).
# This includes the C API, Java API, and protocol buffer srcs.

package(default_visibility = ["//visibility:private"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("@local_config_syslibs//:build_defs.bzl", "if_not_system_lib")
load("//tensorflow:tensorflow.bzl", "tf_binary_additional_srcs")
load("//tensorflow:tensorflow.bzl", "if_cuda")
load("//tensorflow:tensorflow.bzl", "if_not_windows")
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
    name = "libtensorflow_cc",
    extension = "tar.gz",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [
        ":cheaders",
        ":eager_cheaders",
        ":ccheaders",
        ":coreheaders",
        ":clib",
        ":cclib",
        ":clicenses",
        ":cclicenses",
    ],
)

pkg_tar(
    name = "libtensorflow_jni",
    extension = "tar.gz",
    srcs = [
        "include/tensorflow/jni/LICENSE",
        "//:LICENSE",
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
    srcs = tf_binary_additional_srcs(),
    tags = ["manual"],
)

pkg_tar(
    name = "cheaders",
    srcs = [
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
    name = "ccheaders",
    package_dir = "include/tensorflow/cc",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [
        ":client_ccheaders",
        ":framework_ccheaders",
        ":gradient_ccheaders",
        ":ops_ccheaders",
        ":profiler_ccheaders",
        ":saved_model_ccheaders",
        ":tools_ccheaders",
        ":training_ccheaders",
    ],
)

pkg_tar(
    name = "client_ccheaders",
    srcs = [
        "//tensorflow/cc:client_headers",
    ],
    package_dir = "client",
    tags = ["manual"],
)

pkg_tar(
    name = "framework_ccheaders",
    srcs = [
        "//tensorflow/cc:framework_headers",
    ],
    package_dir = "framework",
    tags = ["manual"],
)

pkg_tar(
    name = "gradient_ccheaders",
    srcs = [
        "//tensorflow/cc:gradient_headers",
    ],
    package_dir = "gradient",
    tags = ["manual"],
)

pkg_tar(
    name = "ops_ccheaders",
    srcs = [
        "//tensorflow/cc:ops_headers",
    ],
    package_dir = "ops",
    tags = ["manual"],
)

pkg_tar(
    name = "profiler_ccheaders",
    srcs = [
        "//tensorflow/cc/profiler:headers",
    ],
    package_dir = "profiler",
    tags = ["manual"],
)

pkg_tar(
    name = "saved_model_ccheaders",
    srcs = [
        "//tensorflow/cc/saved_model:headers",
    ],
    package_dir = "saved_model",
    tags = ["manual"],
)

pkg_tar(
    name = "tools_ccheaders",
    srcs = [
        "//tensorflow/cc/tools:headers",
    ],
    package_dir = "tools",
    tags = ["manual"],
)

pkg_tar(
    name = "training_ccheaders",
    srcs = [
        "//tensorflow/cc:training_headers",
    ],
    package_dir = "training",
    tags = ["manual"],
)

pkg_tar(
    name = "coreheaders",
    package_dir = "include/tensorflow/core",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [
        ":framework_coreheaders",
        ":graph_coreheaders",
        ":lib_core_coreheaders",
        ":lib_strings_coreheaders",
        ":platform_coreheaders",
        ":public_coreheaders",
    ],
)

pkg_tar(
    name = "framework_coreheaders",
    srcs = [
        "//tensorflow/core:framework_headers",
    ],
    package_dir = "framework",
    tags = ["manual"],
)

pkg_tar(
    name = "graph_coreheaders",
    srcs = [
        "//tensorflow/core:graph_headers",
    ],
    package_dir = "graph",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_core_coreheaders",
    srcs = [
        "//tensorflow/core:lib_core_headers",
    ],
    package_dir = "lib/core",
    tags = ["manual"],
)

pkg_tar(
    name = "lib_strings_coreheaders",
    srcs = [
        "//tensorflow/core:lib_strings_headers",
    ],
    package_dir = "lib/strings",
    tags = ["manual"],
)

pkg_tar(
    name = "platform_coreheaders",
    srcs = [
        "//tensorflow/core:platform_headers",
    ],
    package_dir = "platform",
    tags = ["manual"],
)

pkg_tar(
    name = "public_coreheaders",
    srcs = [
        "//tensorflow/core:public_headers",
    ],
    package_dir = "public",
    tags = ["manual"],
)

pkg_tar(
    name = "clib",
    srcs = select({
        "//tensorflow:windows": [
            "//tensorflow:tensorflow.dll",
            "//tensorflow:tensorflow_dll_import_lib",
        ],
        "//conditions:default": ["//tensorflow:libtensorflow.so"],
    }),
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
    name = "cclib",
    srcs = select({
        "//tensorflow:windows": [
            "//tensorflow:tensorflow_cc.dll",
            "//tensorflow:tensorflow_cc_dll_import_lib",
        ],
        "//conditions:default": ["//tensorflow:libtensorflow_cc.so"],
    }),
    package_dir = "lib",
    # Mark as "manual" till
    # https://github.com/bazelbuild/bazel/issues/2352
    # and https://github.com/bazelbuild/bazel/issues/1580
    # are resolved, otherwise these rules break when built
    # with Python 3.
    tags = ["manual"],
    deps = [":common_deps"],
)

# pkg_tar(
#     name = "corelib",
#     srcs = select({
#         "//tensorflow:windows": [
#             "//tensorflow:tensorflow_cc.dll",
#             "//tensorflow:tensorflow_cc_dll_import_lib",
#         ],
#         "//conditions:default": ["//tensorflow:libtensorflow_cc.so"],
#     }),
#     package_dir = "lib",
#     # Mark as "manual" till
#     # https://github.com/bazelbuild/bazel/issues/2352
#     # and https://github.com/bazelbuild/bazel/issues/1580
#     # are resolved, otherwise these rules break when built
#     # with Python 3.
#     tags = ["manual"],
#     deps = [":common_deps"],
# )

pkg_tar(
    name = "clicenses",
    srcs = [
        ":include/tensorflow/c/LICENSE",
        "//:LICENSE",
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
    name = "cclicenses",
    srcs = [":include/tensorflow/cc/LICENSE"],
    package_dir = "include/tensorflow/cc",
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
        "@icu//:icu4c/LICENSE",
        "@jpeg//:LICENSE.md",
        "@lmdb//:LICENSE",
        "@local_config_sycl//sycl:LICENSE.text",
        "@nasm//:LICENSE",
        "@nsync//:LICENSE",
        "@png_archive//:LICENSE",
        "@protobuf_archive//:LICENSE",
        "@snappy//:COPYING",
        "@zlib_archive//:zlib.h",
    ] + if_not_windows([
        "@llvm//:LICENSE.TXT",
    ]) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_aws_support": [],
        "//conditions:default": [
            "@aws//:LICENSE",
        ],
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
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
        "@local_config_nccl//:LICENSE",
    ]) + if_mkl([
        "//third_party/mkl:LICENSE",
        "//third_party/mkl_dnn:LICENSE",
    ]) + if_not_system_lib(
        "grpc",
        [
            "@grpc//:LICENSE",
            "@grpc//third_party/address_sorting:LICENSE",
        ],
    ),
    outs = ["include/tensorflow/c/LICENSE"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)

genrule(
    name = "cclicenses_generate",
    srcs = [
        "//third_party/icu/data:LICENSE",
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
        "@icu//:icu4c/LICENSE",
        "@jpeg//:LICENSE.md",
        "@lmdb//:LICENSE",
        "@local_config_sycl//sycl:LICENSE.text",
        "@nasm//:LICENSE",
        "@nsync//:LICENSE",
        "@png_archive//:LICENSE",
        "@protobuf_archive//:LICENSE",
        "@snappy//:COPYING",
        "@zlib_archive//:zlib.h",
    ] + if_not_windows([
        "@llvm//:LICENSE.TXT",
    ]) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_aws_support": [],
        "//conditions:default": [
            "@aws//:LICENSE",
        ],
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
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
        "@local_config_nccl//:LICENSE",
    ]) + if_mkl([
        "//third_party/mkl:LICENSE",
        "//third_party/mkl_dnn:LICENSE",
    ]) + if_not_system_lib(
        "grpc",
        [
            "@grpc//:LICENSE",
            "@grpc//third_party/address_sorting:LICENSE",
        ],
    ),
    outs = ["include/tensorflow/cc/LICENSE"],
    cmd = "$(location :concat_licenses.sh) $(SRCS) >$@",
    tools = [":concat_licenses.sh"],
)

genrule(
    name = "jnilicenses_generate",
    srcs = [
        "//third_party/icu/data:LICENSE",
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
        "@icu//:icu4j/main/shared/licenses/LICENSE",
        "@jpeg//:LICENSE.md",
        "@lmdb//:LICENSE",
        "@local_config_sycl//sycl:LICENSE.text",
        "@nasm//:LICENSE",
        "@nsync//:LICENSE",
        "@png_archive//:LICENSE",
        "@protobuf_archive//:LICENSE",
        "@snappy//:COPYING",
        "@zlib_archive//:zlib.h",
        "@grpc//:LICENSE",
        "@grpc//third_party/address_sorting:LICENSE",
    ] + if_not_windows([
        "@llvm//:LICENSE.TXT",
    ]) + select({
        "//tensorflow:android": [],
        "//tensorflow:ios": [],
        "//tensorflow:linux_s390x": [],
        "//tensorflow:windows": [],
        "//tensorflow:no_aws_support": [],
        "//conditions:default": [
            "@aws//:LICENSE",
        ],
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
    }) + if_cuda([
        "@cub_archive//:LICENSE.TXT",
        "@local_config_nccl//:LICENSE",
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
