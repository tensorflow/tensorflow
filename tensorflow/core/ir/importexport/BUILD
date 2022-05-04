load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/compiler/mlir:glob_lit_test.bzl", "glob_lit_tests")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(
    default_visibility = [
        ":__subpackages__",
        "//tensorflow/core:__subpackages__",
        "//tensorflow/tools/tfg_graph_transforms:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "import",
    srcs = [
        "functiondef_import.cc",
        "import.cc",
    ],
    hdrs = [
        "functiondef_import.h",
        "import.h",
    ],
    visibility = [
        "//tensorflow/compiler/mlir/tensorflow:__subpackages__",
        "//tensorflow/core:__subpackages__",
        "//tensorflow/tools/tfg_graph_transforms:__subpackages__",
    ],
    deps = [
        ":convert_attributes",
        ":convert_tensor",
        ":convert_types",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:function_def_utils",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:path",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "export",
    srcs = [
        "export.cc",
        "functiondef_export.cc",
    ],
    hdrs = [
        "export.h",
        "functiondef_export.h",
    ],
    visibility = [
        "//tensorflow/core:__subpackages__",
        "//tensorflow/tools/tfg_graph_transforms:__subpackages__",
    ],
    deps = [
        ":convert_attributes",
        ":convert_tensor",
        ":convert_types",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
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
        "//tensorflow/stream_executor/lib",
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
        "//tensorflow/stream_executor/lib",
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
    srcs = ["graphdef_import.cc"],
    hdrs = ["graphdef_import.h"],
    deps = [
        ":convert_attributes",
        ":convert_types",
        ":import",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:errors",
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
    srcs = ["graphdef_export.cc"],
    hdrs = ["graphdef_export.h"],
    deps = [
        ":convert_attributes",
        ":convert_types",
        ":export",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/platform:stringpiece",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

# Command line tool to convert to/from GraphDef to MLIR Graph.
tf_cc_binary(
    name = "tfg-translate",
    srcs = ["tfg-translate.cc"],
    deps = [
        ":export",
        ":graphdef_export",
        ":graphdef_import",
        ":import",
        ":load_proto",
        "//tensorflow/core:ops",  # Ops need to be registered for import.
        "//tensorflow/core/ir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
)

glob_lit_tests(
    data = ["//tensorflow/core/ir:test_utilities"],
    driver = "//tensorflow/core/ir:run_lit.sh",
    exclude = [],
    test_file_exts = ["mlir"],
)
