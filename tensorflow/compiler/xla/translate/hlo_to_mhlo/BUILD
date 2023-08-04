load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "attribute_importer",
    srcs = ["attribute_importer.cc"],
    hdrs = ["attribute_importer.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "stack_location_utils",
    srcs = ["stack_location_utils.cc"],
    hdrs = ["stack_location_utils.h"],
    deps = [
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
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
        ":hlo_utils",
        ":location_importer",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:printer",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SparseTensorDialect",
    ],
)

cc_library(
    name = "hlo_to_mlir_hlo",
    srcs = ["hlo_to_mlir_hlo.cc"],
    hdrs = ["hlo_to_mlir_hlo.h"],
    deps = [
        ":hlo_module_importer",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/mlir/utils:error_util",
    ],
)

cc_library(
    name = "hlo_utils",
    srcs = ["hlo_utils.cc"],
    hdrs = ["hlo_utils.h"],
    includes = ["include"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:convert_op_folder",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/service/llvm_ir:llvm_util",
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
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/tsl/platform:protobuf",
        "//tensorflow/tsl/platform:test_main",
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
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "translate",
    srcs = ["translate.cc"],
    hdrs = ["translate.h"],
    deps = [
        ":hlo_to_mlir_hlo",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:mlir_hlo_to_hlo",
        "//tensorflow/tsl/platform:protobuf",
        "@llvm-project//mlir:IR",
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
