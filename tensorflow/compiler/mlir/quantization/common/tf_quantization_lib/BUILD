load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    # By default, these targets should only be used within the quantization library.
    default_visibility = [
        "//learning/brain/mlir/quantization:__subpackages__",
        "//platforms/darwinn/compiler:__subpackages__",
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "tf_quantization_lib",
    srcs = [
        "tf_quantization_driver.cc",
        "tf_quantization_interface.cc.inc",
        "tf_quantization_utils.cc",
    ],
    hdrs = [
        "tf_quantization_driver.h",
        "tf_quantization_interface.h.inc",
        "tf_quantization_traits.h",
        "tf_quantization_utils.h",
    ],
    deps = [
        ":tf_quantization_config",
        ":tf_quantization_interfaces_inc_gen",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/compiler/mlir/lite/quantization/lite/toco_legacy:portable_tensor_utils",
        "//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "//tensorflow/compiler/mlir/tools/optimize:quantization_utils",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "tf_quantization_driver_test",
    srcs = ["tf_quantization_driver_test.cc"],
    deps = [
        ":tf_quantization_lib",
        "//tensorflow/compiler/mlir/quantization/common:attrs_and_constraints",
        "//tensorflow/compiler/mlir/quantization/common:func",
        "//tensorflow/compiler/mlir/quantization/common:test_base",
        "//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tf_quantization_config",
    srcs = [
        "tf_quantization_config.cc",
    ],
    hdrs = [
        "tf_quantization_config.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/lite/tools/optimize:reduced_precision_metadata",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

td_library(
    name = "tf_quantization_td_files",
    srcs = [
        "tf_quantization.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/common/ir:QuantizationOpsTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tf_quantization_interfaces_inc_gen",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {
        "tf_quantization_interface.h.inc": ["-gen-op-interface-decls"],
        "tf_quantization_interface.cc.inc": ["-gen-op-interface-defs"],
    },
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "tf_quantization.td",
    deps = [
        ":tf_quantization_td_files",
    ],
)

exports_files([
    "tf_quantization_traits.h",
    "tf_quantization_config.h",
    "tf_quantization_utils.h",
])
