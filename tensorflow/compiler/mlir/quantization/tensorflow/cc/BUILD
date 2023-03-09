load(
    "//tensorflow:tensorflow.default.bzl",
    "get_compatible_with_cloud",
)
load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    # By default, these targets should only be used within the quantization library.
    default_visibility = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "save_variables",
    srcs = ["save_variables.cc"],
    hdrs = ["save_variables.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:framework",
        "//tensorflow/core/ir/importexport:convert_tensor",
        "//tensorflow/core/util/tensor_bundle",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "save_variables_test",
    srcs = ["save_variables_test.cc"],
    deps = [
        ":save_variables",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/util/tensor_bundle",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:status_matchers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

cc_library(
    name = "const_op_size",
    srcs = ["const_op_size.cc"],
    hdrs = ["const_op_size.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_remaining_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "const_op_size_test",
    srcs = ["const_op_size_test.cc"],
    deps = [
        ":const_op_size",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "convert_asset_args",
    srcs = ["convert_asset_args.cc"],
    hdrs = ["convert_asset_args.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "convert_asset_args_test",
    srcs = ["convert_asset_args_test.cc"],
    deps = [
        ":convert_asset_args",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "status_macro",
    hdrs = ["status_macro.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/tsl/platform:macros",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "status_macro_test",
    srcs = ["status_macro_test.cc"],
    deps = [
        ":status_macro",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/status",
    ],
)
