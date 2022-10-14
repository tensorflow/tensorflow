load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
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
        "//tensorflow/compiler/mlir/xla:location_metadata",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:permutation_util",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/service:computation_layout",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:attribute_importer",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_utils",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
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
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla/mlir/utils:error_util",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_module_importer",
    ],
)

cc_library(
    name = "hlo_utils",
    srcs = ["hlo_utils.cc"],
    hdrs = ["hlo_utils.h"],
    includes = ["include"],
    deps = [
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/mlir_hlo:convert_op_folder",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/tsl/platform:bfloat16",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SparseTensorDialect",
    ],
)

tf_cc_test(
    name = "hlo_utils_test",
    srcs = ["hlo_utils_test.cc"],
    deps = [
        ":hlo_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "//tensorflow/tsl/platform:test_main",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "mlir_hlo_builder",
    srcs = ["mlir_hlo_builder.cc"],
    hdrs = ["mlir_hlo_builder.h"],
    deps = [
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/stream_executor/lib",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:attribute_importer",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_module_importer",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_utils",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "mlir_hlo_builder_test",
    srcs = ["mlir_hlo_builder_test.cc"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_module_importer",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:mlir_hlo_builder",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:test_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)
