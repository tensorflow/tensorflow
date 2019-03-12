# Description:
#   OpKernels and resource templates for lookup tables.

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

cc_library(
    name = "resource_interface_templates",
    hdrs = ["resource_interface_templates.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "op_kernel_templates",
    hdrs = ["op_kernel_templates.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:tensor_flag_utils",
        "//third_party/eigen3",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/types:span",
    ],
)

tf_kernel_library(
    name = "fingerprint64_map_op_kernels",
    srcs = [
        "fingerprint64_map_op_kernels.cc",
    ],
    deps = [
        ":op_kernel_templates",
        ":resource_interface_templates",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
)

tf_kernel_library(
    name = "flat_hash_map_op_kernels",
    srcs = [
        "flat_hash_map_op_kernels.cc",
    ],
    deps = [
        ":op_kernel_templates",
        ":resource_interface_templates",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/kernels:tensor_flag_utils",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

tf_kernel_library(
    name = "generic_table_op_kernels",
    srcs = [
        "generic_table_op_kernels.cc",
    ],
    deps = [
        ":op_kernel_templates",
        ":resource_interface_templates",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:string_view_variant_wrapper",
        "@com_google_absl//absl/strings",
    ],
)
