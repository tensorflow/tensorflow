load("@llvm-project//mlir:tblgen.bzl", "td_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    # By default, these targets should only be used within the quantization library.
    default_visibility = [
        "//learning/brain/mlir/quantization:__subpackages__",
        "//tensorflow/compiler/mlir/lite:__subpackages__",
        "//tensorflow/compiler/mlir/quantization:__subpackages__",
        "//tensorflow/compiler/mlir/stablehlo:__subpackages__",
    ],
    licenses = ["notice"],
)

td_library(
    name = "lift_as_function_call_td_files",
    srcs = [
        "lift_as_function_call.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "lift_as_function_call",
    srcs = ["lift_as_function_call.cc"],
    hdrs = ["lift_as_function_call.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":attrs_and_constraints",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo:stablehlo_type_utils",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:quantization_unit_loc",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:xla_call_module_attrs",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core/ir/types:Dialect",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "lift_as_function_call_test",
    srcs = ["lift_as_function_call_test.cc"],
    deps = [
        ":attrs_and_constraints",
        ":func",
        ":lift_as_function_call",
        ":test_base",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "func",
    srcs = ["func.cc"],
    hdrs = ["func.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/cc/saved_model:signature_constants",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "func_test",
    srcs = ["func_test.cc"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":func",
        ":test_base",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "test_base",
    testonly = 1,
    srcs = [],
    hdrs = ["test_base.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":func",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/lite/quantization/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/common/ir:QuantOps",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc:context",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "attrs_and_constraints",
    srcs = [
        "attrs_and_constraints.cc",
    ],
    hdrs = [
        "attrs_and_constraints.h",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":uniform_quantized_types",
        "//tensorflow/compiler/mlir/quantization/common/quantization_lib",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:xla_call_module_attrs",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

tf_cc_test(
    name = "attrs_and_constraints_test",
    srcs = ["attrs_and_constraints_test.cc"],
    deps = [
        ":attrs_and_constraints",
        ":func",
        ":test_base",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@stablehlo//:stablehlo_ops",
    ],
)

td_library(
    name = "quant_td_files",
    srcs = [
        "attrs_and_constraints.td",
    ],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":lift_as_function_call_td_files",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "uniform_quantized_types",
    srcs = ["uniform_quantized_types.cc"],
    hdrs = ["uniform_quantized_types.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "uniform_quantized_types_test",
    srcs = ["uniform_quantized_types_test.cc"],
    deps = [
        ":test_base",
        ":uniform_quantized_types",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)
