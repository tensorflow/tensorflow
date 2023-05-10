load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_binary", "tf_cc_test")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_compatible_with = get_compatible_with_cloud(),
    default_visibility = [
        "//tensorflow/core:__subpackages__",
        "//tensorflow/tools/tfg_graph_transforms:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

gentbl_cc_library(
    name = "PassIncGen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "--name",
                "TFGraph",
            ],
            "passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "utils",
    srcs = ["utils/utils.cc"],
    hdrs = ["utils/utils.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/ir:Dialect",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "eval_utils",
    srcs = ["utils/eval_utils.cc"],
    hdrs = ["utils/eval_utils.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/importexport:convert_tensor",
        "//tensorflow/core/ir/importexport:graphdef_export",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "PDLLUtilsIncGen",
    tbl_outs = [
        (
            ["-x=cpp"],
            "utils/pdll/PDLLUtils.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-pdll",
    td_file = "utils/pdll/utils.pdll",
    deps = [
        "//tensorflow/core/ir:Dialect",
    ],
)

td_library(
    name = "PDLLUtilsFiles",
    srcs = ["utils/pdll/utils.pdll"],
    includes = ["include"],
)

cc_library(
    name = "pdll_utils",
    srcs = ["utils/pdll/utils.cc"],
    hdrs = ["utils/pdll/utils.h"],
    deps = [
        ":PDLLUtilsIncGen",
        ":utils",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/ir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "op_cat_helper",
    srcs = ["utils/op_cat_helper.cc"],
    hdrs = ["utils/op_cat_helper.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/ir:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "PassRegistration",
    hdrs = ["pass_registration.h"],
    deps = [
        ":PassIncGen",
        "//tensorflow/core/transforms/cf_sink:Pass",
        "//tensorflow/core/transforms/consolidate_attrs:Pass",
        "//tensorflow/core/transforms/const_dedupe_hoist:Pass",
        "//tensorflow/core/transforms/constant_folding:Pass",
        "//tensorflow/core/transforms/cse:Pass",
        "//tensorflow/core/transforms/drop_unregistered_attribute:Pass",
        "//tensorflow/core/transforms/eliminate_passthrough_iter_args:Pass",
        "//tensorflow/core/transforms/func_to_graph:Pass",
        "//tensorflow/core/transforms/functional_to_region:Pass",
        "//tensorflow/core/transforms/graph_compactor:Pass",
        "//tensorflow/core/transforms/graph_to_func:Pass",
        "//tensorflow/core/transforms/legacy_call:Pass",
        "//tensorflow/core/transforms/region_to_functional:Pass",
        "//tensorflow/core/transforms/remapper:Pass",
        "//tensorflow/core/transforms/shape_inference:Pass",
        "//tensorflow/core/transforms/toposort:Pass",
    ],
)

cc_library(
    name = "graph_transform_wrapper",
    srcs = ["graph_transform_wrapper.cc"],
    hdrs = ["graph_transform_wrapper.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core/framework:graph_debug_info_proto_cc",
        "//tensorflow/core/ir/importexport:graphdef_export",
        "//tensorflow/core/ir/importexport:graphdef_import",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "graph_transform_wrapper_test",
    srcs = ["graph_transform_wrapper_test.cc"],
    deps = [
        ":graph_transform_wrapper",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime:graph_constructor",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/platform:status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "eval_utils_test",
    size = "small",
    srcs = ["utils/eval_utils_test.cc"],
    deps = [
        ":eval_utils",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/ir:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

# Custom `mlir-opt` replacement that links our dialect and passes
tf_cc_binary(
    name = "tfg-transforms-opt",
    srcs = ["tfg-transforms-opt.cc"],
    deps = [
        ":PassRegistration",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:ops",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir:tf_op_registry",
        "//tensorflow/core/ir/types:Dialect",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Transforms",
    ],
)

filegroup(
    name = "test_utilities",
    testonly = True,
    data = [
        ":tfg-transforms-opt",
        "@llvm-project//llvm:FileCheck",
        "@llvm-project//llvm:not",
    ],
)
