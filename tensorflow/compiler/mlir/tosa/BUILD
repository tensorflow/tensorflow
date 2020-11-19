# TensorFlow -> TOSA Compiler Bridge.
# See:
#   https://developer.mlplatform.org/w/tosa/
#   https://github.com/llvm/llvm-project/blob/master/mlir/docs/Dialects/TOSA.md

package(
    default_visibility = [":internal"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "internal",
    includes = ["//third_party/mlir:subpackages"],
    packages = [
        "//tensorflow/compiler/mlir/...",
        ],
)

package_group(
    name = "friends",
    includes = [
        ":internal",
    ],
    packages = [
    ],
)

load("//third_party/mlir:tblgen.bzl", "gentbl")

config_setting(
    name = "enable-build",
    values = {"define":"build-tosa=true"},
    visibility = ["//visibility:public"],
)

filegroup(
    name = "tosa_ops_td_files",
    srcs = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:QuantizationOpsTdFiles",
        ],
    )

gentbl(
    name = "tosa_pass_inc_gen",
    tbl_outs = [
        (
            "-gen-pass-decls -name LegalizeTosa",
            "transforms/passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/passes.td",
    td_srcs = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl(
    name = "tosa_legalize_tf_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/tf_legalize_patterns.inc",
        ),
        ],
        tblgen = "@llvm-project//mlir:mlir-tblgen",
        td_file = "transforms/tf_legalize_patterns.td",
        td_srcs = [
            ":tosa_ops_td_files",
            "@llvm-project//mlir:StdOpsTdFiles",
            "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
            ],
    )

gentbl(
    name = "tosa_legalize_tfl_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/tfl_legalize_patterns.inc",
        ),
        ],
        tblgen = "@llvm-project//mlir:mlir-tblgen",
        td_file = "transforms/tfl_legalize_patterns.td",
        td_srcs = [
            ":tosa_ops_td_files",
            "@llvm-project//mlir:StdOpsTdFiles",
            "//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_td_files",
            ],
    )

load("//tensorflow/compiler/mlir/tosa:build_defs.bzl", "cond_cc_library")

cond_cc_library(
    name = "tosa_legalize_tf",
    srcs = [
        "transforms/legalize_tf.cc",
        "transforms/tf_legalize_patterns.inc",
        ],
        hdrs = [
            "transforms/legalize_common.h",
            "transforms/legalize_utils.h",
            "transforms/passes.h",
            "@llvm-project//mlir:include/mlir/Transforms/InliningUtils.h",
            ],
            deps = [
                ":tosa_legalize_tf_inc_gen",
                ":tosa_pass_inc_gen",
                "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
                "//tensorflow/compiler/mlir/tensorflow:translate_lib",
                "//tensorflow/core:lib",
                "//tensorflow/core:framework",
                "//tensorflow/core/kernels:conv_grad_shape_utils",
                "//tensorflow/compiler/mlir/tensorflow:tensorflow_all_ops_inc_gen",
                "@flatbuffers",
                "@com_google_absl//absl/memory",
                "@com_google_absl//absl/container:flat_hash_map",
                "@llvm-project//mlir:Analysis",
                "@llvm-project//mlir:Dialect",
                "@llvm-project//mlir:IR",
                "@llvm-project//mlir:Parser",
                "@llvm-project//mlir:Pass",
                "@llvm-project//mlir:StandardOps",
                "@llvm-project//mlir:TosaDialect",
                "@llvm-project//mlir:Support",
                "@llvm-project//mlir:QuantOps",
                "@llvm-project//mlir:TransformUtils",
                ],
                alwayslink = 1,
    )

cond_cc_library(
    name = "tosa_legalize_tfl",
    srcs = [
        "transforms/legalize_tfl.cc",
        "transforms/tfl_legalize_patterns.inc",
        ],
        hdrs = [
            "transforms/legalize_common.h",
            "transforms/legalize_utils.h",
            "transforms/passes.h",
            "@llvm-project//mlir:include/mlir/Transforms/InliningUtils.h",
            "//tensorflow/compiler/mlir/lite/quantization:quantization_traits.h",
            ],
            deps = [
                ":tosa_legalize_tfl_inc_gen",
                ":tosa_pass_inc_gen",
                "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
                "//tensorflow/compiler/mlir/tensorflow:translate_lib",
                "//tensorflow/core:lib",
                "//tensorflow/core:framework",
                "//tensorflow/core/kernels:conv_grad_shape_utils",
                "//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_inc_gen",
                "//tensorflow/compiler/mlir/lite:validators",
                "//tensorflow/compiler/mlir/lite:tensorflow_lite",
                "//tensorflow/lite/schema:schema_fbs",
                "@flatbuffers",
                "@com_google_absl//absl/memory",
                "@com_google_absl//absl/container:flat_hash_map",
                "@llvm-project//mlir:Analysis",
                "@llvm-project//mlir:Dialect",
                "@llvm-project//mlir:IR",
                "@llvm-project//mlir:Parser",
                "@llvm-project//mlir:Pass",
                "@llvm-project//mlir:StandardOps",
                "@llvm-project//mlir:TosaDialect",
                "@llvm-project//mlir:Support",
                "@llvm-project//mlir:QuantOps",
                "@llvm-project//mlir:TransformUtils",
                ],
                alwayslink = 1,
    )

cond_cc_library(
    name = "tosa_legalize_common",
    srcs = [
        "transforms/legalize_common.cc",
        "transforms/tf_legalize_patterns.inc",
        "transforms/legalize_utils.cc",
        ],
        hdrs = [
            "transforms/legalize_common.h",
            "transforms/legalize_utils.h",
            "@llvm-project//mlir:include/mlir/Transforms/InliningUtils.h",
            ],
            deps = [
                "//tensorflow/compiler/mlir/lite:tensorflow_lite_ops_inc_gen",
                "//tensorflow/compiler/mlir/lite:validators",
                "//tensorflow/compiler/mlir/lite:tensorflow_lite",
                "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
                "//tensorflow/compiler/mlir/tensorflow:translate_lib",
                "//tensorflow/core:lib",
                "//tensorflow/core:framework",
                "//tensorflow/core/kernels:conv_grad_shape_utils",
                "//tensorflow/compiler/mlir/tensorflow:tensorflow_all_ops_inc_gen",
                "@flatbuffers",
                "@com_google_absl//absl/memory",
                "@com_google_absl//absl/container:flat_hash_map",
                "@llvm-project//mlir:Analysis",
                "@llvm-project//mlir:Dialect",
                "@llvm-project//mlir:IR",
                "@llvm-project//mlir:Parser",
                "@llvm-project//mlir:Pass",
                "@llvm-project//mlir:StandardOps",
                "@llvm-project//mlir:TosaDialect",
                "@llvm-project//mlir:Support",
                "@llvm-project//mlir:QuantOps",
                "@llvm-project//mlir:TransformUtils",
                ],
                alwayslink = 1,
    )

cond_cc_library(
    name = "tosa_fuse_bias_tf",
    srcs = [
        "transforms/fuse_bias_tf.cc",
        ],
        hdrs = [
            "transforms/passes.h",
            "@llvm-project//mlir:include/mlir/Transforms/InliningUtils.h",
            ],
            deps = [
                ":tosa_legalize_common",
                ":tosa_pass_inc_gen",
                "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
                "//tensorflow/compiler/mlir/tensorflow:translate_lib",
                "//tensorflow/core:lib",
                "//tensorflow/compiler/mlir/tensorflow:tensorflow_all_ops_inc_gen",
                "@flatbuffers",
                "@com_google_absl//absl/memory",
                "@com_google_absl//absl/container:flat_hash_map",
                "@llvm-project//mlir:Analysis",
                "@llvm-project//mlir:Dialect",
                "@llvm-project//mlir:IR",
                "@llvm-project//mlir:Parser",
                "@llvm-project//mlir:Pass",
                "@llvm-project//mlir:StandardOps",
                "@llvm-project//mlir:TosaDialect",
                "@llvm-project//mlir:Support",
                "@llvm-project//mlir:QuantOps",
                "@llvm-project//mlir:TransformUtils",
                ],
                alwayslink = 1,
    )

cond_cc_library(
    name = "tosa_convert_tfl_uint8",
    srcs = [
        "transforms/convert_tfl_uint8.cc",
        ],
        hdrs = [
            "transforms/passes.h",
            "@llvm-project//mlir:include/mlir/Transforms/InliningUtils.h",
            ],
            deps = [
                ":tosa_legalize_common",
                ":tosa_pass_inc_gen",
                "//tensorflow/compiler/mlir/tensorflow:convert_graphdef",
                "//tensorflow/compiler/mlir/tensorflow:translate_lib",
                "//tensorflow/core:lib",
                "//tensorflow/compiler/mlir/tensorflow:tensorflow_all_ops_inc_gen",
                "@flatbuffers",
                "@com_google_absl//absl/memory",
                "@com_google_absl//absl/container:flat_hash_map",
                "@llvm-project//mlir:Analysis",
                "@llvm-project//mlir:Dialect",
                "@llvm-project//mlir:IR",
                "@llvm-project//mlir:Parser",
                "@llvm-project//mlir:Pass",
                "@llvm-project//mlir:StandardOps",
                "@llvm-project//mlir:TosaDialect",
                "@llvm-project//mlir:Support",
                "@llvm-project//mlir:QuantOps",
                "@llvm-project//mlir:TransformUtils",
                ],
                alwayslink = 1,
    )

cond_cc_library(
    name = "tosa_pipelines",

    srcs = [
        "tosa_passpipes.cc",
        ],
    hdrs = [
          "transforms/passes.h",
          "transforms/register_passes.h",
          "tosa_passpipes.h",
        ],
    deps = [
          ":tosa_pass_inc_gen",
          "@com_google_absl//absl/memory",
          "@com_google_absl//absl/container:flat_hash_map",
          "@llvm-project//mlir:Analysis",
          "@llvm-project//mlir:Dialect",
          "@llvm-project//mlir:IR",
          "@llvm-project//mlir:Parser",
          "@llvm-project//mlir:Pass",
          "@llvm-project//mlir:StandardOps",
          "@llvm-project//mlir:TosaDialect",
          "@llvm-project//mlir:Support",
          "@llvm-project//mlir:QuantOps",
          "@llvm-project//mlir:TransformUtils",
          "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
         ],
         alwayslink = 1,
    )

cond_cc_library(
    name = "tf_tosa_passes",

    srcs = [
        "tf_tosa_pipeline.cc",
        ],
    hdrs = [
        ],
    deps = [
          ":tosa_pass_inc_gen",
          ":tosa_pipelines",
          ":tosa_legalize_tf",
          ":tosa_legalize_common",
          ":tosa_fuse_bias_tf",
          "@com_google_absl//absl/memory",
          "@com_google_absl//absl/container:flat_hash_map",
          "@llvm-project//mlir:Analysis",
          "@llvm-project//mlir:Dialect",
          "@llvm-project//mlir:IR",
          "@llvm-project//mlir:Parser",
          "@llvm-project//mlir:Pass",
          "@llvm-project//mlir:StandardOps",
          "@llvm-project//mlir:TosaDialect",
          "@llvm-project//mlir:Support",
          "@llvm-project//mlir:QuantOps",
          "@llvm-project//mlir:TransformUtils",
          "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
         ],
         alwayslink = 1,
    )

cond_cc_library(
    name = "tfl_tosa_passes",

    srcs = [
        "tfl_tosa_pipeline.cc",
        ],
    hdrs = [
        ],
    deps = [
          ":tosa_pass_inc_gen",
          ":tosa_pipelines",
          ":tosa_legalize_tfl",
          ":tosa_legalize_common",
          ":tosa_convert_tfl_uint8",
          "@com_google_absl//absl/memory",
          "@com_google_absl//absl/container:flat_hash_map",
          "@llvm-project//mlir:Analysis",
          "@llvm-project//mlir:Dialect",
          "@llvm-project//mlir:IR",
          "@llvm-project//mlir:Parser",
          "@llvm-project//mlir:Pass",
          "@llvm-project//mlir:StandardOps",
          "@llvm-project//mlir:TosaDialect",
          "@llvm-project//mlir:Support",
          "@llvm-project//mlir:QuantOps",
          "@llvm-project//mlir:TransformUtils",
          "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
         ],
         alwayslink = 1,
    )
