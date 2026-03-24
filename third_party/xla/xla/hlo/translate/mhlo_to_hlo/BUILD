load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")
load("//xla/tsl/platform:rules_cc.bzl", "cc_binary", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/mlir:tensorflow_friends",
        "//learning/brain/mlir:xla_friends",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "attribute_exporter",
    srcs = ["attribute_exporter.cc"],
    hdrs = ["attribute_exporter.h"],
    deps = [
        ":type_to_shape",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_utils",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:unregistered_attributes",
        "//xla/service:hlo_proto_cc",
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
        "//xla/service/spmd/shardy/stablehlo_round_trip:export_shardings",
        "//xla/stream_executor:dnn",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:base",
        "@stablehlo//:stablehlo_ops",
    ],
)

xla_cc_test(
    name = "attribute_exporter_test",
    srcs = ["attribute_exporter_test.cc"],
    deps = [
        ":attribute_exporter",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo_sharding",
        "//xla/hlo/parser:hlo_parser",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo",
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "//xla/tsl/util/proto:proto_matchers",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:register",
    ],
)

cc_library(
    name = "layout_util",
    srcs = ["layout_util.cc"],
    hdrs = ["layout_util.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/ir:hlo",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cc_library(
    name = "literal_exporter",
    srcs = ["literal_exporter.cc"],
    hdrs = ["literal_exporter.h"],
    deps = [
        ":type_to_shape",
        "//xla:array",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "location_exporter",
    srcs = ["location_exporter.cc"],
    hdrs = ["location_exporter.h"],
    deps = [
        ":stack_frame_index_builder",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_utils",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "module_attributes_exporter",
    srcs = ["module_attributes_exporter.cc"],
    hdrs = ["module_attributes_exporter.h"],
    deps = [
        "//xla:xla_data_proto_cc",
        "//xla/mlir_hlo:unregistered_attributes",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "stack_frame_index_builder",
    srcs = ["stack_frame_index_builder.cc"],
    hdrs = ["stack_frame_index_builder.h"],
    deps = [
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/strings:string_view",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "mlir_hlo_to_hlo",
    srcs = [
        "hlo_op_writer.inc",
        "mlir_hlo_to_hlo.cc",
    ],
    hdrs = ["mlir_hlo_to_hlo.h"],
    deps = [
        ":attribute_exporter",
        ":hlo_op_writer_inc",
        ":layout_util",
        ":literal_exporter",
        ":location_exporter",
        ":module_attributes_exporter",
        ":stack_frame_index_builder",
        ":type_to_shape",
        "//xla:array",
        "//xla:comparison_util",
        "//xla:debug_options_flags",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/builder/lib:approx_topk",
        "//xla/hlo/builder/lib:approx_topk_shape",
        "//xla/hlo/builder/lib:math",
        "//xla/hlo/builder/lib:matrix",
        "//xla/hlo/builder/lib:quantize",
        "//xla/hlo/builder/lib:slicing",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_utils",
        "//xla/mlir/utils:error_util",
        "//xla/mlir/utils:type_util",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/mlir_hlo:stablehlo_extension_passes",
        "//xla/mlir_hlo:unregistered_attributes",
        "//xla/service:collective_ops_utils",
        "//xla/service:computation_layout",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/service:source_target_pairs",
        "//xla/service/gpu:backend_configs_cc",
        "//xla/service/spmd/shardy:constants",
        "//xla/service/spmd/shardy:utils",
        "//xla/service/spmd/shardy/stablehlo_round_trip:export_shardings",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@shardy//shardy/dialect/sdy/ir:dialect",
        "@stablehlo//:base",
        "@stablehlo//:stablehlo_ops",
        "@stablehlo//:stablehlo_passes",
        "@tsl//tsl/platform:ml_dtypes",
    ],
)

build_test(
    name = "gen_hlo_op_writer_build_test",
    targets = [
        ":gen_hlo_op_writer",
    ],
)

cc_binary(
    name = "gen_hlo_op_writer",
    srcs = ["gen_hlo_op_writer.cc"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//mlir:TableGen",
    ],
)

gentbl_cc_library(
    name = "hlo_op_writer_inc",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = {"hlo_op_writer.inc": []},
    tblgen = ":gen_hlo_op_writer",
    td_file = "gen_hlo_op_writer.td",
    deps = [
        "//xla/mlir_hlo:hlo_ops_td_files",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@stablehlo//:stablehlo_ops_td_filegroup",
    ],
)

xla_cc_test(
    name = "mlir_hlo_to_hlo_test",
    srcs = ["mlir_hlo_to_hlo_test.cc"],
    deps = [
        ":mlir_hlo_to_hlo",
        "//xla/hlo/translate:register",
        "//xla/mlir/utils:error_util",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@com_google_absl//absl/status:status_matchers",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

xla_cc_test(
    name = "stack_frame_index_builder_test",
    srcs = ["stack_frame_index_builder_test.cc"],
    deps = [
        ":stack_frame_index_builder",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:test",
        "//xla/tsl/platform:test_main",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "translate",
    srcs = ["translate.cc"],
    hdrs = ["translate.h"],
    deps = [
        ":attribute_exporter",
        ":mlir_hlo_to_hlo",
        ":type_to_shape",
        "//xla:debug_options_flags",
        "//xla:shape_util",
        "//xla/hlo/builder:xla_builder",
        "//xla/hlo/builder:xla_computation",
        "//xla/hlo/ir:hlo",
        "//xla/hlo/translate:register",
        "//xla/mlir_hlo:unregistered_attributes",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_proto_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "translate_registration",
    testonly = True,
    srcs = [
        "translate_registration.cc",
        "translate_registration.h",
    ],
    deps = [
        ":translate",
        "//xla/hlo/translate:register",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "type_to_shape",
    srcs = ["type_to_shape.cc"],
    hdrs = ["type_to_shape.h"],
    deps = [
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/mlir/utils:type_util",
        "//xla/mlir_hlo",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:SparseTensorEnums",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

xla_cc_test(
    name = "type_to_shape_test",
    srcs = ["type_to_shape_test.cc"],
    deps = [
        ":type_to_shape",
        "//xla:shape_util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/testlib:test",
        "//xla/hlo/translate/hlo_to_mhlo:hlo_utils",
        "//xla/mlir_hlo",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@stablehlo//:stablehlo_ops",
        "@tsl//tsl/platform:protobuf",
    ],
)
