load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    features = [
        # Required since headers are not self-contained.
        "-parse_headers",
    ],
)

config_setting(
    # Add "--define tensorflow_mkldnn_contraction_kernel=0" to your build command to disable mkldnn
    # sgemm in Eigen tensor contractions (matrix multiplications and convolutions). The mkldnn
    # kernels are generated at runtime and use avx/avx2/fma/avx512 based on cpu status registers
    # (https://en.wikipedia.org/wiki/CPUID). Default Eigen contraction kernel is
    # Eigen::internal::gebp_kernel (general block-panel kernel).
    #
    # NOTE: Prefer using the :onednn_contraction_kernel flag or the
    # cc_binary_disable_onednn build rule.
    name = "no_mkldnn_contraction_kernel",
    define_values = {
        "tensorflow_mkldnn_contraction_kernel": "0",
    },
)

bzl_library(
    name = "build_defs",
    srcs = ["build_defs.bzl"],
)

# Add
# `--//third_party/tensorflow/compiler/xla/tsl/framework/contraction:disable_onednn_contraction_kernel=True`
# or use the `cc_binary_disable_onednn` build rule to disable the oneDNN
# library for tensor contractions (new name of the MKL-DNN library).
bool_flag(
    name = "disable_onednn_contraction_kernel",
    build_setting_default = False,
)

config_setting(
    name = "disable_onednn_contraction_kernel_config",
    flag_values = {":disable_onednn_contraction_kernel": "True"},
)

# Depending on a build configuration this target provides custom kernel for Eigen
# tensor contractions (small matrix multiplication kernel used to multiple together
# blocks of the original tensors).
#
# 1) Default:
#    Use Mkldnn single threaded sgemm. The mkldnn kernels are generated at runtime and
#    use avx/avx2/fma/avx512 based on cpu status registers (https://en.wikipedia.org/wiki/CPUID).
#
# 2) Eigen: --define tensorflow_mkldnn_contraction_kernel=0 (disable mkldnn)
#    Use Eigen contraction kernel: Eigen::internal::gebp_kernel.
#
# If you use `tensor.contract(other_tensor)` in your code, you must include additional header
# to get the benefit of custom contraction kernel:
#
#   #if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#   #include "third_party/tensorflow/compiler/xla/tsl/framework/contraction/eigen_contraction_kernel.h"
#   #endif
#
# We define a two-level target because if we just add
#   ":no_mkldnn_contraction_kernel": []
# in the same select list with //third_party/tensorflow:{android,arm,ios,ppc},
# there can be more than one match, e.g., when building for android and MKL-DNN
# contraction kernel is disabled. Bazel doesn't allow multiple matches.
# See more details in
#   https://github.com/tensorflow/tensorflow/issues/24414
cc_library(
    name = "eigen_contraction_kernel",
    hdrs = ["eigen_contraction_kernel.h"],
    compatible_with = get_compatible_with_portable(),
    deps = select({
        ":no_mkldnn_contraction_kernel": [":eigen_contraction_kernel_no_mkl"],
        ":disable_onednn_contraction_kernel_config": [":eigen_contraction_kernel_no_mkl"],
        "//conditions:default": [":eigen_contraction_kernel_with_mkl"],
    }) + ["@com_google_absl//absl/base"],
)

cc_library(
    name = "eigen_contraction_kernel_with_mkl",
    srcs = ["eigen_contraction_kernel.cc"],
    hdrs = ["eigen_contraction_kernel.h"],
    defines = select({
        ":disable_onednn_contraction_kernel_config": [],
        "//xla/tsl:android_x86": [],
        "//xla/tsl:arm_any": [
            "TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL",
        ],
        "//xla/tsl:fuchsia_x86_64": [],
        "//xla/tsl:ios": [],
        "//xla/tsl:linux_ppc64le": [],
        "//xla/tsl:linux_s390x": [],
        "//xla/tsl:macos_arm64": [],
        "//conditions:default": [
            "TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL",
            "TENSORFLOW_USE_MKLDNN_CONTRACTION_KERNEL",
        ],
    }),
    deps = [
        "//xla/tsl/framework/fixedpoint",
        "@com_google_absl//absl/base",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:dynamic_annotations",
    ] + select({
        ":disable_onednn_contraction_kernel_config": [],
        "//xla/tsl:android_x86": [],
        "//xla/tsl:arm_any": [],
        "//xla/tsl:fuchsia_x86_64": [],
        "//xla/tsl:ios": [],
        "//xla/tsl:linux_ppc64le": [],
        "//xla/tsl:linux_s390x": [],
        "//xla/tsl:macos_arm64": [],
        "//conditions:default": ["@onednn//:mkl_dnn"],
    }),
)

# Portable Tensorflow for Android/iOS requires these files directly rather than as libraries, so
# export them to be used there.
exports_files(
    srcs = [
        "eigen_contraction_kernel.cc",
        "eigen_contraction_kernel.h",
    ],
)

cc_library(
    name = "eigen_contraction_kernel_no_mkl",
    srcs = ["eigen_contraction_kernel.cc"],
    hdrs = ["eigen_contraction_kernel.h"],
    compatible_with = get_compatible_with_portable(),
    # Somehow the following code works with fixedpoint, but not here.
    # visibility = [
    #     "//tensorflow:__subpackages__",
    #     "//xla/tsl:internal",
    # ],
    deps = [
        "//xla/tsl/framework/fixedpoint",
        "@com_google_absl//absl/base",
        "@eigen_archive//:eigen3",
        "@local_tsl//tsl/platform:dynamic_annotations",
    ],
)

# Maintain the same name as other directories until a principled refactor is done, as these files
# used to all be a single target.
filegroup(
    name = "xla_cpu_runtime_hdrs",
    srcs = [
        "eigen_contraction_kernel.h",
    ],
)

# Maintain the same name as other directories until a principled refactor is done, as these files
# used to all be a single target.
filegroup(
    name = "xla_cpu_runtime_srcs",
    srcs = [
        "eigen_contraction_kernel.cc",
    ],
)
