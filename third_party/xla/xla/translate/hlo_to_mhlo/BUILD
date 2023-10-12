load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("//xla:xla.bzl", "xla_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "attribute_importer",
    srcs = ["attribute_importer.cc"],
    hdrs = ["attribute_importer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:shape_util",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/mlir_hlo",
        "//xla/service:hlo_proto_cc",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "stack_location_utils",
    srcs = ["stack_location_utils.cc"],
    hdrs = ["stack_location_utils.h"],
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
    deps = [
        ":attribute_importer",
        ":hlo_utils",
        ":location_importer",
        "//xla:comparison_util",
        "//xla:permutation_util",
        "//xla:printer",
        "//xla:protobuf_util",
        "//xla:shape_util",
        "//xla:status",
        "//xla:status_macros",
        "//xla:statusor",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla:xla_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SparseTensorDialect",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "hlo_to_mlir_hlo",
    srcs = ["hlo_to_mlir_hlo.cc"],
    hdrs = ["hlo_to_mlir_hlo.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":hlo_module_importer",
        "//xla:status",
        "//xla:status_macros",
        "//xla/mlir/utils:error_util",
    ],
)

cc_library(
    name = "hlo_utils",
    srcs = ["hlo_utils.cc"],
    hdrs = ["hlo_utils.h"],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "//xla:literal",
        "//xla:shape_util",
        "//xla:types",
        "//xla:util",
        "//xla:xla_data_proto_cc",
        "//xla/hlo/ir:hlo",
        "//xla/mlir_hlo",
        "//xla/mlir_hlo:convert_op_folder",
        "//xla/mlir_hlo:lhlo",
        "//xla/service/llvm_ir:llvm_util",
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
        "@local_tsl//tsl/platform:protobuf",
        "@local_tsl//tsl/platform:test_main",
    ],
)

cc_library(
    name = "location_importer",
    srcs = ["location_importer.cc"],
    hdrs = ["location_importer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "stack_location_utils",
        "//xla/hlo/ir:hlo",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "translate",
    srcs = ["translate.cc"],
    hdrs = ["translate.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":hlo_to_mlir_hlo",
        "//xla:status",
        "//xla/mlir_hlo",
        "//xla/service:hlo_parser",
        "//xla/service:hlo_proto_cc",
        "//xla/service/llvm_ir:llvm_util",
        "//xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:protobuf",
    ],
)

cc_library(
    name = "translate_registration",
    testonly = True,
    srcs = ["translate_registration.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":translate",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TranslateLib",
    ],
    alwayslink = 1,
)
