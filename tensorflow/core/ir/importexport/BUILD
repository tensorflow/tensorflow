load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = [
        ":__subpackages__",
        "//tensorflow/compiler/mlir:__subpackages__",
        "//tensorflow/core:__subpackages__",
        "//tensorflow/tools/tfg_graph_transforms:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "parse_text_proto",
    srcs = [
        "parse_text_proto.cc",
    ],
    hdrs = ["parse_text_proto.h"],
    deps = [
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/platform:casts",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "mangling",
    srcs = [
        "mangling.cc",
    ],
    hdrs = ["mangling.h"],
    deps = [
        ":parse_text_proto",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "load_proto",
    srcs = [
        "load_proto.cc",
    ],
    hdrs = ["load_proto.h"],
    deps = [
        ":parse_text_proto",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib_proto_parsing",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "convert_types",
    srcs = [
        "convert_types.cc",
    ],
    hdrs = ["convert_types.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "convert_tensor",
    srcs = [
        "convert_tensor.cc",
    ],
    hdrs = ["convert_tensor.h"],
    deps = [
        ":convert_types",
        ":mangling",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:tflite_portable_logging",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "savedmodel_import",
    srcs = ["savedmodel_import.cc"],
    hdrs = ["savedmodel_import.h"],
    deps = [
        ":graphdef_import",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "savedmodel_export",
    srcs = ["savedmodel_export.cc"],
    hdrs = ["savedmodel_export.h"],
    deps = [
        ":graphdef_export",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "graphdef_import",
    srcs = [
        "functiondef_import.cc",
        "graphdef_import.cc",
    ],
    hdrs = [
        "functiondef_import.h",
        "graphdef_import.h",
    ],
    deps = [
        ":convert_attributes",
        ":convert_types",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:stringpiece",
        "@com_google_absl//absl/container:flat_hash_map",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "graphdef_export",
    srcs = [
        "functiondef_export.cc",
        "graphdef_export.cc",
    ],
    hdrs = [
        "functiondef_export.h",
        "graphdef_export.h",
    ],
    deps = [
        ":convert_attributes",
        ":convert_types",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "convert_attributes",
    srcs = [
        "convert_attributes.cc",
    ],
    hdrs = ["convert_attributes.h"],
    deps = [
        ":convert_tensor",
        ":convert_types",
        ":mangling",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

# Command line tool to convert to/from GraphDef to MLIR Graph.
tf_cc_binary(
    name = "tfg-translate",
    srcs = ["tfg-translate.cc"],
    deps = [
        ":graphdef_export",
        ":graphdef_import",
        ":load_proto",
        "//tensorflow/core:ops",  # Ops need to be registered for import.
        "//tensorflow/core/ir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
)

glob_lit_tests(
    data = ["//tensorflow/core/ir:test_utilities"],
    driver = "//tensorflow/core/ir:run_lit.sh",
    exclude = [],
    test_file_exts = ["mlir"],
)
