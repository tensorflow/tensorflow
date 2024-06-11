load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([
        "//learning/brain/mlir:tensorflow_friends",
        "//learning/brain/mlir:xla_friends",
    ]),
    licenses = ["notice"],
)

cc_library(
    name = "attribute_importer",
    srcs = ["attribute_importer.cc"],
    hdrs = ["attribute_importer.h"],
    deps = [
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/mlir_hlo",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "custom_call_importer",
    srcs = ["custom_call_importer.cc"],
    hdrs = ["custom_call_importer.h"],
    deps = [
        "//xla:statusor",
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "stack_location_utils",
    srcs = ["stack_location_utils.cc"],
    hdrs = ["stack_location_utils.h"],
    deps = [
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "hlo_module_importer",
    srcs = [
        "hlo_function_importer.cc",
        "hlo_module_importer.cc",
    ],
    hdrs = [
        "hlo_function_importer.h",
        "hlo_module_importer.h",
    ],
    deps = [
        ":attribute_importer",
        ":custom_call_importer",
        ":hlo_utils",
        ":location_importer",
        ":module_config_importer",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:protobuf_util",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/service:hlo_module_config",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:errors",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_to_mlir_hlo",
    srcs = ["hlo_to_mlir_hlo.cc"],
    hdrs = ["hlo_to_mlir_hlo.h"],
    deps = [
        ":hlo_module_importer",
        "//xla:status_macros",
        "//xla/mlir/utils:error_util",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "hlo_utils",
    srcs = ["hlo_utils.cc"],
    hdrs = ["hlo_utils.h"],
    includes = ["include"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/mlir/utils:type_util",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:convert_op_folder",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:SparseTensorEnums",
    ],
)

xla_cc_test(
    name = "hlo_utils_test",
    srcs = ["hlo_utils_test.cc"],
    deps = [
        ":hlo_utils",
        "//xla:literal",
        "//xla:literal_util",
        "//xla:shape_util",
        "//xla:test",
        "//xla:types",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "location_importer",
    srcs = ["location_importer.cc"],
    hdrs = ["location_importer.h"],
    deps = [
        "stack_location_utils",
        "//xla/hlo/ir:hlo",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "module_config_importer",
    srcs = ["module_config_importer.cc"],
    hdrs = ["module_config_importer.h"],
    deps = [
        "//xla/service:hlo_module_config",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "translate",
    srcs = ["translate.cc"],
    hdrs = ["translate.h"],
    deps = [
        ":hlo_to_mlir_hlo",
        "//xla/mlir_hlo",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "translate_registration",
    testonly = True,
    srcs = ["translate_registration.cc"],
    deps = [
        ":translate",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TranslateLib",
    ],
    alwayslink = 1,
)
