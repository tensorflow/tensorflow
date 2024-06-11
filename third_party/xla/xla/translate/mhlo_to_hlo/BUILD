load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")
load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_binary", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl:tsl.default.bzl", "get_compatible_with_portable")

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
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/mlir_hlo",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_proto_cc",
        "//xla/stream_executor:dnn",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "layout_util",
    srcs = ["layout_util.cc"],
    hdrs = ["layout_util.h"],
    deps = [
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/hlo/ir:hlo",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "location_exporter",
    srcs = ["location_exporter.cc"],
    hdrs = ["location_exporter.h"],
    deps = [
        ":stack_frame_index_builder",
        "//xla:xla_data_proto_cc",
        "@com_google_absl//absl/log",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "module_config_exporter",
    srcs = ["module_config_exporter.cc"],
    hdrs = ["module_config_exporter.h"],
    deps = [
        "//xla/service:hlo_module_config",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "stack_frame_index_builder",
    srcs = ["stack_frame_index_builder.cc"],
    hdrs = ["stack_frame_index_builder.h"],
    deps = [
        "//xla/service:hlo_proto_cc",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "mlir_hlo_to_hlo",
    srcs = [
        "mlir_hlo_to_hlo.cc",
        "operator_writers.inc",
    ],
    hdrs = ["mlir_hlo_to_hlo.h"],
    deps = [
        ":attribute_exporter",
        ":layout_util",
        ":location_exporter",
        ":module_config_exporter",
        ":operator_writer_inc",
        ":stack_frame_index_builder",
        ":type_to_shape",
        "//xla:array",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/client:xla_builder",
        "//xla/client/lib:approx_topk",
        "//xla/client/lib:approx_topk_shape",
        "//xla/client/lib:matrix",
        "//xla/client/lib:quantize",
        "//xla/client/lib:slicing",
        "//xla/hlo/ir:hlo",
        "//xla/mlir/utils:error_util",
        "//xla/mlir/utils:type_util",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_proto_cc",
        "//xla/service/gpu:backend_configs_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_tsl//tsl/platform:statusor",
        "@stablehlo//:stablehlo_ops",
    ],
)

build_test(
    name = "operator_writer_gen_build_test",
    targets = [
        ":operator_writer_gen",
    ],
)

cc_binary(
    name = "operator_writer_gen",
    srcs = ["operator_writer_gen.cc"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//mlir:TableGen",
    ],
)

gentbl_cc_library(
    name = "operator_writer_inc",
    compatible_with = get_compatible_with_portable(),
    tbl_outs = [([], "operator_writers.inc")],
    tblgen = ":operator_writer_gen",
    td_file = "//xla/mlir_hlo:mhlo/IR/hlo_ops.td",
    deps = [
        "//xla/mlir_hlo:hlo_ops_td_files",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

cc_library(
    name = "translate",
    srcs = ["translate.cc"],
    hdrs = ["translate.h"],
    deps = [
        ":mlir_hlo_to_hlo",
        ":type_to_shape",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "//xla/service:hlo_proto_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "translate_registration",
    testonly = True,
    srcs = ["translate_registration.cc"],
    deps = [
        ":translate",
        "//xla/mlir_hlo:hlo_dialect_registration",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TensorDialect",
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
        "//xla:test",
        "//xla:xla_data_proto_cc",
        "//xla/translate/hlo_to_mhlo:hlo_utils",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:test_main",
    ],
)
