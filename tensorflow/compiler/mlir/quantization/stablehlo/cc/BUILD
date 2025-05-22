load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load(
    "//tensorflow:tensorflow.default.bzl",
    "get_compatible_with_portable",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # For TFLite Converter integration.
        "//tensorflow/compiler/mlir/lite:__subpackages__",
        "//tensorflow/compiler/mlir/quantization:__subpackages__",
        "//tensorflow/compiler/mlir/tensorflow_to_stablehlo:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "types",
    hdrs = ["types.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [],
)

cc_library(
    name = "component",
    hdrs = ["component.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "config",
    srcs = ["config.cc"],
    hdrs = ["config.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings:string_view",
    ],
)

tf_cc_test(
    name = "config_test",
    srcs = ["config_test.cc"],
    deps = [
        ":config",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "io",
    srcs = ["io.cc"],
    hdrs = ["io.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/strings:string_view",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "io_test",
    srcs = ["io_test.cc"],
    deps = [
        ":io",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/tsl/platform:env",
        "@local_xla//xla/tsl/platform:status",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "graph_def",
    srcs = [],
    hdrs = ["graph_def.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/core:protos_all_cc",
    ],
)

tf_cc_test(
    name = "graph_def_test",
    srcs = ["graph_def_test.cc"],
    deps = [
        ":graph_def",
        "//tensorflow/core:protos_all_cc",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "debugger",
    srcs = ["debugger.cc"],
    hdrs = ["debugger.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_quant_ops",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "permutation",
    hdrs = ["permutation.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "permutation_test",
    srcs = ["permutation_test.cc"],
    deps = [
        ":permutation",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tf_saved_model_export",
    srcs = ["tf_saved_model_export.cc"],
    hdrs = ["tf_saved_model_export.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":io",
        ":tf_pass_pipeline",
        ":tf_saved_model_import",
        ":types",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:convert_asset_args",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:tf_unfreeze_constants",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow/transforms:verify_no_outside_compilation_markers_pass",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "tf_saved_model_export_test",
    srcs = ["tf_saved_model_export_test.cc"],
    deps = [
        ":tf_saved_model_export",
        "//tensorflow/compiler/mlir/quantization/common:tf_test_base",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/core:all_kernels",  # buildcleaner: keep Required to export to GraphDef
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:ops",  # buildcleaner: keep Required to export to GraphDef
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "saved_model_export",
    srcs = ["saved_model_export.cc"],
    hdrs = ["saved_model_export.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":io",
        ":pass_pipeline",
        ":saved_model_import",
        ":types",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:convert_asset_args",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:unfreeze_constants",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow/transforms:verify_no_outside_compilation_markers_pass",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "saved_model_export_test",
    srcs = ["saved_model_export_test.cc"],
    deps = [
        ":saved_model_export",
        "//tensorflow/compiler/mlir/quantization/common:test_base",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/core:all_kernels",  # buildcleaner: keep Required to export to GraphDef
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:ops",  # buildcleaner: keep Required to export to GraphDef
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/tsl/platform:status_matchers",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "tf_saved_model_import",
    srcs = ["tf_saved_model_import.cc"],
    hdrs = ["tf_saved_model_import.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":types",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:reader",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_quantize_preprocess",
        "//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "tf_saved_model_import_test",
    srcs = ["tf_saved_model_import_test.cc"],
    deps = [
        ":tf_saved_model_import",
        ":types",
        "//tensorflow/compiler/mlir/quantization/common:tf_test_base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "saved_model_import",
    srcs = ["saved_model_import.cc"],
    hdrs = ["saved_model_import.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":types",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:reader",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantize_preprocess",
        "//tensorflow/compiler/mlir/tensorflow:mlir_import_options",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

tf_cc_test(
    name = "saved_model_import_test",
    srcs = ["saved_model_import_test.cc"],
    deps = [
        ":saved_model_import",
        ":types",
        "//tensorflow/compiler/mlir/quantization/common:test_base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_pass_pipeline",
    srcs = ["tf_pass_pipeline.cc"],
    hdrs = ["tf_pass_pipeline.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:bridge_passes",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo:tf_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_passes",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@stablehlo//:stablehlo_passes",
    ],
)

cc_library(
    name = "pass_pipeline",
    srcs = ["pass_pipeline.cc"],
    hdrs = ["pass_pipeline.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:bridge_passes",
        "//tensorflow/compiler/mlir/quantization/stablehlo:passes",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ReconcileUnrealizedCasts",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@stablehlo//:stablehlo_passes",
    ],
)

cc_library(
    name = "tf_pre_calibration",
    srcs = ["tf_pre_calibration.cc"],
    hdrs = ["tf_pre_calibration.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:__subpackages__",
        "//tensorflow/compiler/mlir/quantization/tensorflow:__subpackages__",
    ],
    deps = [
        ":component",
        ":tf_pass_pipeline",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "pre_calibration",
    srcs = ["pre_calibration.cc"],
    hdrs = ["pre_calibration.h"],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/compiler/mlir/quantization/stablehlo:__subpackages__",
        "//tensorflow/compiler/mlir/quantization/tensorflow:__subpackages__",
    ],
    deps = [
        ":component",
        ":pass_pipeline",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

tf_cc_test(
    name = "pre_calibration_test",
    srcs = ["pre_calibration_test.cc"],
    deps = [
        ":config",
        ":pre_calibration",
        "//tensorflow/compiler/mlir/quantization/common:test_base",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_quant_ops",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_xla//xla/tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "tf_report",
    srcs = ["tf_report.cc"],
    hdrs = ["tf_report.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":io",
        "//tensorflow/compiler/mlir/quantization/common:tf_lift_as_function_call",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "report",
    srcs = ["report.cc"],
    hdrs = ["report.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":io",
        "//tensorflow/compiler/mlir/quantization/common:lift_as_function_call",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

tf_cc_test(
    name = "report_test",
    srcs = ["report_test.cc"],
    deps = [
        ":io",
        ":report",
        "//tensorflow/compiler/mlir/quantization/common:test_base",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:protobuf",
        "@local_xla//xla/tsl/platform:status_matchers",
    ],
)

cc_library(
    name = "context",
    srcs = [],
    hdrs = ["context.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_post_calibration",
    srcs = ["tf_post_calibration.cc"],
    hdrs = ["tf_post_calibration.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":component",
        ":config",
        ":tf_pass_pipeline",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/instrumentations:tf_save_report",
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "post_calibration",
    srcs = ["post_calibration.cc"],
    hdrs = ["post_calibration.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":component",
        ":config",
        ":pass_pipeline",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/instrumentations:save_report",
        "//tensorflow/compiler/mlir/quantization/tensorflow:passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "static_range_ptq",
    srcs = ["static_range_ptq.cc"],
    hdrs = ["static_range_ptq.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":component",
        ":context",
        ":tf_post_calibration",
        ":tf_pre_calibration",
        ":tf_saved_model_export",
        ":tf_saved_model_import",
        ":types",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration:component",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:quantization_options_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "tf_weight_only_ptq",
    srcs = ["tf_weight_only_ptq.cc"],
    hdrs = ["tf_weight_only_ptq.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":component",
        ":config",
        ":context",
        ":tf_pass_pipeline",
        ":tf_saved_model_export",
        ":tf_saved_model_import",
        ":types",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/instrumentations:tf_save_report",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:tf_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)

cc_library(
    name = "weight_only_ptq",
    srcs = ["weight_only_ptq.cc"],
    hdrs = ["weight_only_ptq.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        ":component",
        ":config",
        ":context",
        ":pass_pipeline",
        ":saved_model_export",
        ":saved_model_import",
        ":types",
        "//tensorflow/compiler/mlir/quantization/stablehlo:quantization_config_proto_cc",
        "//tensorflow/compiler/mlir/quantization/stablehlo/instrumentations:save_report",
        "//tensorflow/compiler/mlir/quantization/tensorflow:exported_model_proto_cc",
        "//tensorflow/compiler/mlir/quantization/tensorflow:passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/cc:run_passes",
        "//tensorflow/compiler/mlir/quantization/tensorflow/python:py_function_lib",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/base:nullability",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla/mlir_hlo:mhlo_passes",
        "@local_xla//xla/tsl/platform:errors",
        "@local_xla//xla/tsl/platform:statusor",
    ],
)
