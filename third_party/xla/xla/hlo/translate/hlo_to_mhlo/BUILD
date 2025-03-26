load("//xla:xla.default.bzl", "xla_cc_test")
load("//xla/tsl:tsl.bzl", "internal_visibility")
load("//xla/tsl/platform:rules_cc.bzl", "cc_library")

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
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "async_importer",
    srcs = ["async_importer.cc"],
    hdrs = ["async_importer.h"],
    deps = [
        ":attribute_importer",
        ":hlo_utils",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/tsl/platform:errors",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "custom_call_importer",
    srcs = ["custom_call_importer.cc"],
    hdrs = ["custom_call_importer.h"],
    deps = [
        "//xla:util",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
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
        ":hlo_utils",
        "//xla/hlo/ir:hlo",
        "//xla/service:hlo_proto_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "hlo_function_importer",
    srcs = ["hlo_function_importer.cc"],
    hdrs = ["hlo_function_importer.h"],
    deps = [
        ":async_importer",
        ":attribute_importer",
        ":custom_call_importer",
        ":hlo_utils",
        ":location_importer",
        "//xla:comparison_util",
        "//xla:literal",
        "//xla:protobuf_util",
        "//xla:shape_util",
        "//xla:status_macros",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/service:hlo_proto_cc",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:SparseTensorDialect",
        "@llvm-project//mlir:Support",
        "@stablehlo//:base",
    ],
)

cc_library(
    name = "hlo_module_importer",
    srcs = [
        "hlo_module_importer.cc",
    ],
    hdrs = [
        "hlo_module_importer.h",
    ],
    deps = [
        ":hlo_function_importer",
        ":module_attributes_importer",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
    ],
)

cc_library(
    name = "hlo_to_mlir_hlo",
    srcs = ["hlo_to_mlir_hlo.cc"],
    hdrs = ["hlo_to_mlir_hlo.h"],
    deps = [
        ":hlo_module_importer",
        "//xla/mlir/utils:error_util",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/tsl/platform:errors",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
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
        "//xla/mlir/utils:type_util",
        "//xla/mlir_hlo",
        "//xla/tsl/platform:statusor",
        "@com_google_absl//absl/status:statusor",
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
        "//xla:types",
        "//xla:xla_data_proto_cc",
        "//xla/mlir_hlo",
        "//xla/tsl/lib/core:status_test_util",
        "//xla/tsl/platform:statusor",
        "//xla/tsl/platform:test_main",
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "location_importer",
    srcs = ["location_importer.cc"],
    hdrs = ["location_importer.h"],
    deps = [
        "stack_location_utils",
        ":hlo_utils",
        "//xla/hlo/ir:hlo",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "module_attributes_importer",
    srcs = ["module_attributes_importer.cc"],
    hdrs = ["module_attributes_importer.h"],
    deps = [
        ":hlo_function_importer",
        ":hlo_utils",
        "//xla:shape_layout",
        "//xla:shape_util",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/service:computation_layout",
        "//xla/service:hlo_module_config",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "translate",
    srcs = ["translate.cc"],
    hdrs = ["translate.h"],
    deps = [
        ":hlo_to_mlir_hlo",
        "//xla/hlo/parser:hlo_parser",
        "//xla/hlo/translate:stablehlo",
        "//xla/mlir_hlo:mhlo_passes",
        "//xla/service:hlo_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "translate_registration",
    testonly = True,
    srcs = ["translate_registration.cc"],
    deps = [
        ":translate",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TranslateLib",
    ],
    alwayslink = 1,
)
