load("//tensorflow:tensorflow.default.bzl", "filegroup", "get_compatible_with_cloud")
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_gen_op_wrapper_py")

# copybara:uncomment_begin(google-only)
# load("//learning/brain/experimental/mlir/tensorflow/dialectgen:dialectgen.bzl", "dialectgen")
#
# copybara:uncomment_end(google-only)
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files([
    "ir/tf_generated_ops.td",
    "ir/tf_op_base.td",
    "ir/tf_op_interfaces.td",
    "ir/tf_ops.td",
])

td_library(
    name = "tensorflow_ops_td_files",
    srcs = [
        "ir/tf_generated_ops.td",
        "ir/tf_op_base.td",
        "ir/tf_op_interfaces.td",
        "ir/tf_ops.td",
        "ir/tfrt_ops.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@llvm-project//mlir:CallInterfacesTdFiles",
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:LoopLikeInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "tensorflow_op_interfaces_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-interface-decls"],
            "ir/tf_op_interfaces.h.inc",
        ),
        (
            ["-gen-op-interface-defs"],
            "ir/tf_op_interfaces.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_op_interfaces.td",
    test = True,
    deps = [
        ":tensorflow_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tensorflow_struct_doc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-dialect-doc"],
            "g3doc/tf_ops.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_ops.td",
    test = True,
    deps = [
        ":tensorflow_ops_td_files",
    ],
)

cc_library(
    name = "tensorflow_op_interfaces",
    srcs = [
        "ir/tf_op_interfaces.cc",
        "ir/tf_op_interfaces.cc.inc",
        "ir/tf_op_interfaces.h.inc",
        "ir/tf_verifiers.cc",
    ],
    hdrs = [
        "ir/tf_op_interfaces.h",
        "ir/tf_verifiers.h",
    ],
    deps = [
        ":tensorflow_op_interfaces_inc_gen",
        ":tensorflow_structs",
        "//tensorflow/core:framework",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "tensorflow_all_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ir/tf_all_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ir/tf_all_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_ops.td",
    deps = [
        ":tensorflow_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tensorflow_tfrt_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ir/tfrt_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ir/tfrt_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tfrt_ops.td",
    deps = [
        ":tensorflow_ops_td_files",
    ],
)

# We only shard tf_op on name for build performance reasons.
tf_ops_category_list = [
    {
        "name": "ops_a_m",
        "include": "tf.[A-M].*$",
    },
    {
        "name": "ops_n_z",
        "include": "tf.[N-Z].*$",
    },
]

[[
    gentbl_cc_library(
        name = "tensorflow_" + target["name"] + "_inc_gen",
        compatible_with = get_compatible_with_cloud(),
        tbl_outs = [
            (
                [
                    "-gen-op-decls",
                    "-op-include-regex=" + target["include"],
                ],
                "ir/tf_" + target["name"] + ".h.inc",
            ),
            (
                [
                    "-gen-op-defs",
                    "-op-include-regex=" + target["include"],
                ],
                "ir/tf_" + target["name"] + ".cc.inc",
            ),
        ],
        tblgen = "@llvm-project//mlir:mlir-tblgen",
        td_file = "ir/tf_ops.td",
        deps = [
            ":tensorflow_ops_td_files",
        ],
    ),
] for target in tf_ops_category_list]

gentbl_cc_library(
    name = "tensorflow_remaining_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-op-decls",
                "-op-exclude-regex=" + "|".join([target["include"] for target in tf_ops_category_list]),
            ],
            "ir/tf_remaining_ops.h.inc",
        ),
        (
            [
                "-gen-op-defs",
                "-op-exclude-regex=" + "|".join([target["include"] for target in tf_ops_category_list]),
            ],
            "ir/tf_remaining_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_ops.td",
    deps = [
        ":tensorflow_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tf_saved_model_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ir/tf_saved_model.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ir/tf_saved_model.cc.inc",
        ),
        (
            ["-gen-dialect-doc"],
            "g3doc/tf_saved_model.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_saved_model_ops.td",
    test = True,
    deps = [
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tensorflow_executor_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ir/tf_executor.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ir/tf_executor.cc.inc",
        ),
        (
            [
                "-gen-dialect-doc",
                "-dialect=tf_executor",
            ],
            "g3doc/tf_executor.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_executor_ops.td",
    test = True,
    deps = [
        ":tensorflow_ops_td_files",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:InferTypeOpInterfaceTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tensorflow_device_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "ir/tf_device.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "ir/tf_device.cc.inc",
        ),
        (
            ["-gen-dialect-doc"],
            "g3doc/tf_device.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_device_ops.td",
    test = True,
    deps = [
        "@llvm-project//mlir:ControlFlowInterfacesTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
    ],
)

gentbl_cc_library(
    name = "tensorflow_canonicalize_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_canonicalize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/canonicalize.td",
    deps = [
        ":rewrite_util_td_files",
        ":tensorflow_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tensorflow_reduce_patterns_inc_gen",
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/reducer/tf_reduce_patterns.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/reducer/tf_mlir_reduce_patterns.td",
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "hlo_legalize_tf_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_legalize_hlo.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_hlo_patterns.td",
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
        "//tensorflow/compiler/xla/mlir_hlo:hlo_ops_td_files",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "tf_legalize_hlo",
    srcs = [
        "transforms/generated_legalize_hlo.inc",
        "transforms/legalize_hlo.cc",
        "transforms/passes.h",
    ],
    deps = [
        ":lower_tf_lib",
        ":tensorflow",
        ":tf_device_pass_inc_gen",
        ":tf_pass_inc_gen",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@stablehlo//:broadcast_utils",
        "@stablehlo//:chlo_ops",
    ],
)

cc_library(
    name = "tfe_legalize_tfg",
    srcs = [
        "transforms/passes.h",
        "transforms/tfg-to-tfe.cc",
    ],
    deps = [
        ":tensorflow",
        ":tf_device_pass_inc_gen",
        ":tf_pass_inc_gen",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/transforms/toposort:Pass",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "mlprogram",
    srcs = [
        "transforms/mlprogram.cc",
    ],
    hdrs = [
        "transforms/mlprogram.h",
    ],
    deps = [
        ":tensorflow_passes",
        ":tf_saved_model_passes",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "//tensorflow/compiler/mlir/tf2xla:legalize_tf",
        "//tensorflow/compiler/xla/mlir_hlo:mhlo_passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "tensorflow_attributes",
    hdrs = [
        "ir/tf_attributes.h",
        "ir/tf_dialect.h",
    ],
    deps = [
        ":tensorflow_types",
        "//tensorflow/core/ir/types:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tensorflow_traits",
    srcs = [
    ],
    hdrs = [
        "ir/tf_traits.h",
    ],
    deps = [
        ":tensorflow_op_interfaces",
        ":tensorflow_types",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

# TensorFlow ops are separated into `tensorflow_ops_a_m.cc` and
# `tensorflow_ops_n_z.cc` so that C++ compiler won't be stressed by huge C++
# files. However, there might be dependencies between `tensorflow_ops_a_m.cc`
# and `tensorflow_ops_n_z.cc`, thus they must be built in one `cc_library`.
cc_library(
    name = "tensorflow_ops_sharded",
    srcs = [
               "ir/tf_dialect.h",
               "ir/tf_ops.h",
               "ir/tfrt_ops.h",
               "ir/tf_remaining_ops.h",
           ] + ["ir/tf_" + target["name"] + ".cc" for target in tf_ops_category_list] +
           ["ir/tf_" + target["name"] + ".cc.inc" for target in tf_ops_category_list] +
           ["ir/tf_" + target["name"] + ".h" for target in tf_ops_category_list],
    hdrs = [
    ],
    textual_hdrs = [
        "ir/tf_types.def",
        "ir/tf_all_ops.h.inc",
        "ir/tfrt_ops.h.inc",
        "ir/tf_remaining_ops.h.inc",
    ] + ["ir/tf_" + target["name"] + ".h.inc" for target in tf_ops_category_list],
    deps = [
        ":attribute_utils",
        ":dynamic_shape_utils",
        ":tensorflow_attributes",
        ":rewrite_util",
        ":tensorflow_canonicalize_inc_gen",
        ":tensorflow_op_interfaces",
        ":tensorflow_op_interfaces_inc_gen",
        ":tensorflow_side_effects",
        ":tensorflow_structs",
        ":tensorflow_traits",
        ":tensorflow_types",
        ":tf_ops_canonicalization_helper",
        ":tf_ops_device_helper",
        ":tf_ops_layout_helper",
        ":tf_ops_tensor_helper",
        ":tf_arith_ops_folder",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Support",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ] + [":tensorflow_" + target["name"] + "_inc_gen" for target in tf_ops_category_list],
)

cc_library(
    name = "tensorflow_remaining_ops",
    srcs = [
        "ir/tf_dialect.h",
        "ir/tf_ops.h",
        "ir/tf_remaining_ops.h",
        "ir/tf_remaining_ops.cc",
        "ir/tfrt_ops.h",
    ] + ["ir/tf_" + target["name"] + ".h" for target in tf_ops_category_list],
    hdrs = [
    ],
    textual_hdrs = [
        "ir/tf_all_ops.h.inc",
        "ir/tf_remaining_ops.h.inc",
        "ir/tfrt_ops.h.inc",
    ] + ["ir/tf_" + target["name"] + ".h.inc" for target in tf_ops_category_list],
    deps = [
        ":rewrite_util",
        ":serialize_mlir_module_utils",
        ":tensorflow_attributes",
        ":tensorflow_canonicalize_inc_gen",
        ":tensorflow_op_interfaces",
        ":tensorflow_op_interfaces_inc_gen",
        ":tensorflow_remaining_ops_inc_gen",
        ":tensorflow_side_effects",
        ":tensorflow_structs",
        ":tensorflow_tfrt_ops_inc_gen",
        ":tensorflow_traits",
        ":tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tensorflow_tfrt_ops",
    srcs = [
        "ir/tf_dialect.h",
        "ir/tf_ops.h",
        "ir/tfrt_ops.h",
        "ir/tfrt_ops.cc",
        "ir/tf_remaining_ops.h",
    ] + ["ir/tf_" + target["name"] + ".h" for target in tf_ops_category_list],
    hdrs = [
    ],
    textual_hdrs = [
        "ir/tf_all_ops.h.inc",
        "ir/tfrt_ops.h.inc",
        "ir/tf_remaining_ops.h.inc",
    ] + ["ir/tf_" + target["name"] + ".h.inc" for target in tf_ops_category_list],
    deps = [
        ":rewrite_util",
        ":tensorflow_attributes",
        ":tensorflow_canonicalize_inc_gen",
        ":tensorflow_op_interfaces",
        ":tensorflow_op_interfaces_inc_gen",
        ":tensorflow_remaining_ops_inc_gen",
        ":tensorflow_side_effects",
        ":tensorflow_structs",
        ":tensorflow_tfrt_ops_inc_gen",
        ":tensorflow_traits",
        ":tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/framework:resource_handle",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tensorflow_ops",
    srcs = [
        "ir/tf_dialect.h",
        "ir/tf_ops.cc",
        "ir/tf_ops.h",
    ],
    textual_hdrs = [
        "ir/tf_all_ops.h.inc",
        "ir/tf_remaining_ops.h",
        "ir/tfrt_ops.h",
    ] + ["ir/tf_" + target["name"] + ".h" for target in tf_ops_category_list],
    deps = [
        ":rewrite_util",
        ":tensorflow_all_ops_inc_gen",
        ":tensorflow_attributes",
        ":tensorflow_canonicalize_inc_gen",
        ":tensorflow_op_interfaces",
        ":tensorflow_op_interfaces_inc_gen",
        ":tensorflow_ops_sharded",
        ":tensorflow_remaining_ops",
        ":tensorflow_remaining_ops_inc_gen",
        ":tensorflow_side_effects",
        ":tensorflow_structs",
        ":tensorflow_tfrt_ops",
        ":tensorflow_tfrt_ops_inc_gen",
        ":tensorflow_traits",
        ":tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:inline_function_utils",
        "//tensorflow/core/common_runtime:lower_function_call_inline_policy",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tensorflow_structs",
    srcs = [
        "ir/tf_structs.cc",
    ],
    hdrs = [
        "ir/tf_structs.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/ir/types:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tensorflow_side_effects",
    srcs = [
    ],
    hdrs = [
        "ir/tf_side_effects.h",
    ],
    deps = ["@llvm-project//mlir:SideEffectInterfaces"],
)

cc_library(
    name = "tensorflow_types",
    hdrs = [
        "ir/tf_dialect.h",
        "ir/tf_types.h",
    ],
    textual_hdrs = [
        "ir/tf_types.def",
    ],
    deps = [
        "//tensorflow/core/ir/types:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tensorflow",
    srcs = [
        "ir/tf_device.cc",
        "ir/tf_executor.cc",
        "ir/tf_executor.cc.inc",
        "ir/tf_executor.h.inc",
        "ir/tf_saved_model.cc",
        "transforms/tf_device_passes.h.inc",
    ],
    hdrs = [
        "dialect_registration.h",
        "ir/tf_device.h",
        "ir/tf_dialect.h",
        "ir/tf_executor.h",
        "ir/tf_ops.h",
        "ir/tf_saved_model.h",
        "ir/tf_structs.h",
        "@llvm-project//mlir:include/mlir/Interfaces/CallInterfaces.h",
        "@llvm-project//mlir:include/mlir/Transforms/InliningUtils.h",
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensorflow_all_ops_inc_gen",
        ":tensorflow_attributes",
        ":tensorflow_canonicalize_inc_gen",
        ":tensorflow_device_ops_inc_gen",
        ":tensorflow_executor_inc_gen",
        ":tensorflow_op_interfaces",
        ":tensorflow_ops",
        ":tensorflow_side_effects",
        ":tensorflow_structs",
        ":tensorflow_tfrt_ops_inc_gen",
        ":tensorflow_traits",
        ":tensorflow_types",
        ":tf_saved_model_inc_gen",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:logging",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:CallOpInterfacesIncGen",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:MLProgramDialect",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

gentbl_cc_library(
    name = "decompose_resource_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_decompose_resource_ops.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/decompose_resource_ops.td",
    deps = [
        ":rewrite_util_td_files",
        ":tensorflow_ops_td_files",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

tf_cc_test(
    name = "tf_saved_model_test",
    srcs = ["ir/tf_saved_model_test.cc"],
    deps = [
        ":tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:test",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "decompose_resource_ops",
    srcs = [
        "transforms/decompose_resource_ops.cc",
    ],
    hdrs = [
        "transforms/decompose_resource_ops.h",
    ],
    deps = [
        ":decompose_resource_ops_inc_gen",
        ":rewrite_util",
        ":tensorflow",
        ":tensorflow_types",
        "//tensorflow/core:framework",
        "@llvm-project//mlir:IR",
    ],
)

td_library(
    name = "rewrite_util_td_files",
    srcs = [
        "transforms/rewrite_util.td",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "rewrite_util",
    srcs = [
        "transforms/rewrite_util.cc",
    ],
    hdrs = [
        "transforms/rewrite_util.h",
    ],
    deps = [
        "//tensorflow/core:framework",
        "@llvm-project//mlir:IR",
    ],
)

gentbl_cc_library(
    name = "tf_data_optimization_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_tf_data_optimization.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tf_data_optimization.td",
    deps = [
        ":tensorflow_ops_td_files",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "tf_data_optimization",
    srcs = [
        "transforms/tf_data_optimization.cc",
    ],
    hdrs = [
        "transforms/tf_data_optimization.h",
    ],
    deps = [
        ":tensorflow",
        ":tensorflow_types",
        ":tf_data_optimization_inc_gen",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "unroll_batch_matmul_pass",
    srcs = [
        "transforms/unroll_batch_matmul.cc",
    ],
    hdrs = [
        "transforms/unroll_batch_matmul.h",
    ],
    deps = [
        ":tensorflow",
        ":tf_pass_inc_gen",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "lift_variables_lib",
    srcs = [
        "transforms/lift_variables.cc",
    ],
    hdrs = [
        "transforms/lift_variables.h",
    ],
    deps = [
        ":convert_tensor",
        ":tensorflow",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:threadpool_options",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "mark_initialized_variables_lib",
    srcs = [
        "transforms/mark_initialized_variables.cc",
    ],
    hdrs = [
        "transforms/mark_initialized_variables.h",
    ],
    deps = [
        ":session_utils",
        ":tensorflow_ops",
        "//tensorflow/compiler/mlir/utils:string_container_utils",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "fake_session",
    srcs = ["utils/fake_session.cc"],
    hdrs = ["utils/fake_session.h"],
    deps = [
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:threadpool_device",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:threadpool_options",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "session_utils",
    srcs = ["utils/session_utils.cc"],
    hdrs = ["utils/session_utils.h"],
    deps = [
        ":tensorflow",
        ":tensorflow_ops",
        "//tensorflow/compiler/mlir/utils:string_container_utils",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_saved_model_freeze_variables",
    srcs = [
        "transforms/tf_saved_model_freeze_variables.cc",
    ],
    hdrs = [
        "transforms/tf_saved_model_freeze_variables.h",
    ],
    deps = [
        ":convert_tensor",
        ":resource_value_typed_analyzer",
        ":session_utils",
        ":tensorflow",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "topological_sort",
    srcs = ["utils/topological_sort.cc"],
    hdrs = ["utils/topological_sort.h"],
    deps = [
        "@com_google_absl//absl/types:span",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "initialize_variables_in_session_init",
    srcs = [
        "transforms/initialize_variables_in_session_init.cc",
    ],
    hdrs = [
        "transforms/initialize_variables_in_session_init.h",
    ],
    deps = [
        ":convert_tensor",
        ":session_utils",
        ":tensorflow",
        ":tensorflow_ops",
        ":tensorflow_passes",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework_internal",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "tf_saved_model_passes",
    srcs = [
        "transforms/convert_session_initializer_to_function.cc",
        "transforms/deduplicate_bound_input_bindings.cc",
        "transforms/freeze_global_tensors.cc",
        "transforms/freeze_saved_model_assets.cc",
        "transforms/lower_globals_to_ml_program.cc",
        "transforms/lower_variable_ops_to_ml_program.cc",
        "transforms/optimize_global_tensors.cc",
        "transforms/remove_vars_in_session_initializer.cc",
        "transforms/strip_saved_module_metadata.cc",
    ],
    hdrs = [
        "transforms/tf_saved_model_passes.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":resource_value_typed_analyzer",
        ":tensorflow",
        ":tensorflow_analysis",
        ":tensorflow_ops",
        ":tensorflow_passes",
        ":tensorflow_types",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MLProgramDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "tensorflow_analysis",
    srcs = [
        "analysis/per_function_aggregate_analysis.h",
        "analysis/resource_alias_analysis.cc",
        "analysis/resource_dataflow.cc",
        "analysis/side_effect_analysis.cc",
    ],
    hdrs = [
        "analysis/resource_alias_analysis.h",
        "analysis/resource_dataflow.h",
        "analysis/side_effect_analysis.h",
    ],
    deps = [
        ":tensorflow",
        ":tensorflow_op_interfaces",
        ":tensorflow_side_effects",
        ":tensorflow_types",
        "@com_google_absl//absl/container:node_hash_map",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

gentbl_cc_library(
    name = "tf_pass_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorFlow",
            ],
            "transforms/tf_passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "g3doc/_includes/tf_passes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tf_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tf_device_pass_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorFlowDevice",
            ],
            "transforms/tf_device_passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "g3doc/includes/tf_device_passes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tf_device_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tf_savedmodel_pass_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorFlowSavedModel",
            ],
            "transforms/tf_savedmodel_passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "g3doc/includes/tf_savedmodel_passes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tf_savedmodel_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "tf_test_passes_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=TensorFlowTest",
            ],
            "transforms/test_passes.h.inc",
        ),
        (
            ["-gen-pass-doc"],
            "g3doc/includes/tf_test_passes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tf_test_passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "tensorflow_passes",
    srcs = [
        "transforms/add_functions_for_exported_names.cc",
        "transforms/annotate_parameter_replication.cc",
        "transforms/batchmatmul_to_einsum.cc",
        "transforms/bridge.cc",
        "transforms/canonicalize_compile_and_replicate_attributes.cc",
        "transforms/check_control_dependencies.cc",
        "transforms/cluster_formation.cc",
        "transforms/cluster_ops_by_policy.cc",
        "transforms/cluster_outlining.cc",
        "transforms/cluster_tf_ops_pass.cc",
        "transforms/collection_ops_util.cc",
        "transforms/constant_op_device_assignment.cc",
        "transforms/convert_control_to_data_outputs.cc",
        "transforms/convert_launch_func_to_tf_call.cc",
        "transforms/convert_tf_control_flow_to_scf.cc",
        "transforms/convert_to_legacy_compile_and_replicate_attributes.cc",
        "transforms/decompose_reduce_dataset.cc",
        "transforms/decompose_resource_ops_pass.cc",
        "transforms/device_attribute_to_launch.cc",
        "transforms/device_index_selector.cc",
        "transforms/drop_while_shape_invariant.cc",
        "transforms/einsum.cc",
        "transforms/executor_island_coarsening.cc",
        "transforms/executor_tpuv1_inline_tpu_island.cc",
        "transforms/executor_tpuv1_island_coarsening.cc",
        "transforms/executor_tpuv1_outline_tpu_island.cc",
        "transforms/extract_head_tail_outside_compilation.cc",
        "transforms/extract_outside_compilation.cc",
        "transforms/extract_tpu_copy_with_dynamic_shape_op.cc",
        "transforms/fold_broadcast.cc",
        "transforms/functional_control_flow_to_cfg.cc",
        "transforms/functional_control_flow_to_regions.cc",
        "transforms/fused_kernel_matcher.cc",
        "transforms/generated_canonicalize.inc",
        "transforms/generated_optimize.inc",
        "transforms/gpu_fusion.cc",
        "transforms/graph_pruning.cc",
        "transforms/group_by_dialect.cc",
        "transforms/guarantee_all_funcs_one_use.cc",
        "transforms/hoist_loop_invariant.cc",
        "transforms/hoist_replicate_invariant_resource_writes.cc",
        "transforms/host_launch_to_outside_compiled.cc",
        "transforms/init_text_file_to_import.cc",
        "transforms/launch_to_device_attribute.cc",
        "transforms/layout_optimization.cc",
        "transforms/localize_var_handles.cc",
        "transforms/lower_quantized.cc",
        "transforms/mark_input_output_aliases.cc",
        "transforms/mark_ops_for_outside_compilation.cc",
        "transforms/materialize_mlir_passthrough_op.cc",
        "transforms/merge_control_flow.cc",
        "transforms/name_anonymous_iterators.cc",
        "transforms/optimize.cc",
        "transforms/order_by_dialect.cc",
        "transforms/outside_compiled_to_host_launch.cc",
        "transforms/parallel_execute_to_islands.cc",
        "transforms/prepare_tpu_computation_for_tf_export.cc",
        "transforms/promote_resources_to_args.cc",
        "transforms/readonly_references_to_resources.cc",
        "transforms/region_control_flow_to_functional.cc",
        "transforms/remove_unused_arguments.cc",
        "transforms/remove_unused_while_results.cc",
        "transforms/replica_id_to_device_ordinal.cc",
        "transforms/replicate_invariant_op_hoisting.cc",
        "transforms/replicate_tensor_list_init_ops_pass.cc",
        "transforms/replicate_to_island.cc",
        "transforms/resource_device_inference.cc",
        "transforms/resource_op_lifting.cc",
        "transforms/resource_op_lifting_cleanup.cc",
        "transforms/resource_op_lifting_cleanup.h",
        "transforms/rewrite_tpu_embedding_ops.cc",
        "transforms/sink_constant.cc",
        "transforms/stack_ops_decomposition.cc",
        "transforms/strip_noinline_attribute.cc",
        "transforms/strip_tf_attributes.cc",
        "transforms/tensor_array_ops_decomposition.cc",
        "transforms/tensor_device_copy_conversion.cc",
        "transforms/tensor_list_ops_decomposition.cc",
        "transforms/test_resource_alias_analysis.cc",
        "transforms/tf_data_optimization_pass.cc",
        "transforms/tf_device_assignment.cc",
        "transforms/tpu_annotate_dynamic_shape_inputs.cc",
        "transforms/tpu_cluster_cleanup_attributes.cc",
        "transforms/tpu_cluster_formation.cc",
        "transforms/tpu_colocate_composite_resource_ops.cc",
        "transforms/tpu_device_propagation.cc",
        "transforms/tpu_dynamic_layout_pass.cc",
        "transforms/tpu_host_computation_expansion.cc",
        "transforms/tpu_identity_pruning.cc",
        "transforms/tpu_merge_variables_with_execute.cc",
        "transforms/tpu_parallel_execute_sink_resource_write.cc",
        "transforms/tpu_partitioned_op_conversion.cc",
        "transforms/tpu_reorder_replicate_and_partitioned_inputs.cc",
        "transforms/tpu_resource_partitioning.cc",
        "transforms/tpu_resource_read_for_write.cc",
        "transforms/tpu_rewrite_pass.cc",
        "transforms/tpu_sharding_identification_pass.cc",
        "transforms/tpu_space_to_depth_pass.cc",
        "transforms/tpu_update_embedding_enqueue_op_inputs.cc",
        "transforms/tpu_validate_inputs.cc",
        "transforms/tpu_variable_runtime_reformatting.cc",
        "transforms/update_control_dependencies.cc",
        "transforms/verify_suitable_for_graph_export_pass.cc",
        "transforms/xla_cluster_formation.cc",
        "transforms/xla_inline_device_ops.cc",
        "transforms/xla_rewrite.cc",
        "translate/breakup-islands.cc",
        "translate/split_into_island_per_op_pass.cc",
        "translate/tf_executor_to_functional.cc",
        "translate/tf_functional_to_executor.cc",
    ],
    hdrs = [
        "transforms/bridge.h",
        "transforms/call_graph_util.h",
        "transforms/cluster_ops_by_policy.h",
        "transforms/collection_ops_util.h",
        "transforms/einsum.h",
        "transforms/passes.h",
        "translate/split_into_island_per_op_pass.h",
    ],
    includes = ["include"],
    textual_hdrs = [
        "transforms/tf_device_passes.h.inc",
        "transforms/tf_passes.h.inc",
        "transforms/tf_savedmodel_passes.h.inc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":attribute_utils",
        ":bridge_logger",
        ":cluster_util",
        ":convert_tensor",
        ":convert_type",
        ":decompose_resource_ops",
        ":decompose_resource_ops_inc_gen",
        ":device_util",
        ":dump_mlir_util",
        ":dynamic_shape_utils",
        ":error_util",
        ":export_tf_dialect_op",
        ":lower_tf_lib",
        ":mangling_util",
        ":parallel_execute_util",
        ":serialize_mlir_module_utils",
        ":shape_inference_pass",
        ":tensorflow",
        ":tensorflow_analysis",
        ":tensorflow_ops",
        ":tensorflow_optimize_inc_gen",
        ":tensorflow_side_effects",
        ":tensorflow_types",
        ":tf_data_optimization",
        ":tf_legalize_hlo",
        ":tf_ops_layout_helper",
        ":tf_pass_inc_gen",
        ":tf_savedmodel_pass_inc_gen",
        ":tfe_legalize_tfg",
        ":topological_sort",
        ":tpu_cluster_util",
        ":tpu_embedding_ops_registry",
        ":tpu_rewrite_device_util",
        ":translate_utils",
        ":unroll_batch_matmul_pass",
        ":verification_utils",
        ":verify_suitable_for_graph_export",
        ":xla_sharding_util",
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/mlir/lite:validators",
        "//tensorflow/compiler/mlir/tf2xla:xla_legalize_tf",
        "//tensorflow/compiler/mlir/tf2xla:xla_legalize_tf_with_tf2xla",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/client:sharding_builder",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:error_payloads",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:random",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/protobuf/tpu:dynamic_padding_proto_cc",
        "//tensorflow/core/tpu:tpu_embedding_optimization_parameters_utils",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:variant",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AffineAnalysis",
        "@llvm-project//mlir:AffineUtils",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "shape_inference_pass",
    srcs = [
        "transforms/passes.h",
        "transforms/shape_inference.cc",
        "transforms/shape_inference_pass.cc",
    ],
    hdrs = [
        "transforms/shape_inference.h",
    ],
    deps = [
        ":dynamic_shape_utils",
        ":serialize_mlir_module_utils",
        ":shape_inference_utils",
        ":tensorflow",
        ":tensorflow_types",
        ":tf_device_pass_inc_gen",
        ":tf_pass_inc_gen",
        ":translate_utils",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:window_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:shape_inference",
        "//tensorflow/compiler/xla/translate/hlo_to_mhlo:hlo_utils",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir/types:Dialect",
        "@com_google_absl//absl/container:flat_hash_set",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "bridge_pass_test_pipeline_registration",
    testonly = True,  # Ensure alwayslink does not leak in the codebase.
    srcs = [
        "transforms/bridge_pass.cc",
    ],
    deps = [
        ":error_util",
        ":tensorflow_passes",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_test_passes",
    testonly = True,  # Ensure alwayslink does not leak in the codebase.
    srcs = [
        "transforms/init_text_file_to_import_test_pass.cc",
        "transforms/initialize_variables_in_session_init_test_pass.cc",
        "transforms/lift_variables_test_pass.cc",
        "transforms/lower_tf_test_pass.cc",
        "transforms/mark_initialized_variables_test_pass.cc",
        "transforms/resource_analyzer_test_pass.cc",
        "transforms/test_cluster_ops_by_policy.cc",
        "transforms/test_passes.h.inc",
        "transforms/test_side_effect_analysis.cc",
        "transforms/tf_saved_model_freeze_variables_test_pass.cc",
    ],
    hdrs = [
        "transforms/test_passes.h",
    ],
    deps = [
        ":error_util",
        ":fake_session",
        ":initialize_variables_in_session_init",
        ":lift_variables_lib",
        ":lower_tf_lib",
        ":mark_initialized_variables_lib",
        ":resource_value_typed_analyzer",
        ":tensorflow",
        ":tensorflow_analysis",
        ":tensorflow_passes",
        ":tf_saved_model_freeze_variables",
        ":tf_saved_model_passes",
        ":tf_test_passes_inc_gen",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:threadpool_options",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "graph_optimization_pass",
    srcs = ["transforms/graph_optimization_pass.cc"],
    hdrs = ["transforms/graph_optimization_pass.h"],
    deps = [
        ":dump_mlir_util",
        ":error_util",
        ":tensorflow_passes",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "graph_optimization_pass_registration",
    srcs = ["transforms/graph_optimization_pass_registration.cc"],
    deps = [
        ":graph_optimization_pass",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass_registration",
    ],
)

cc_library(
    name = "upgrade_graph",
    srcs = ["translate/upgrade_graph.cc"],
    hdrs = ["translate/upgrade_graph.h"],
    deps = [
        ":attribute_utils",
        "//tensorflow/compiler/tf2xla:functionalize_control_flow",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:grappler_item_builder",
        "//tensorflow/core/grappler/clusters:virtual_cluster",
        "//tensorflow/core/grappler/optimizers:meta_optimizer",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "export_graphdef",
    srcs = [
        "translate/export_graphdef.cc",
    ],
    hdrs = [
        "translate/export_graphdef.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":convert_type",
        ":error_util",
        ":export_tf_dialect_op",
        ":export_utils",
        ":mlir_roundtrip_flags",
        ":tensorflow",
        ":translate_utils",
        ":verify_suitable_for_graph_export",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/mlir/utils:name_utils",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/graph/regularization:util",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "import_model",
    srcs = [
        "translate/import_model.cc",
    ],
    hdrs = [
        "translate/export_graphdef.h",
        "translate/import_model.h",
    ],
    deps = [
        ":convert_attr",
        ":convert_tensor",
        ":convert_type",
        ":dump_mlir_util",
        ":dynamic_shape_utils",
        ":error_util",
        ":initialize_variables_in_session_init",
        ":lift_variables_lib",
        ":mangling_util",
        ":mark_initialized_variables_lib",
        ":mlir_import_options",
        ":mlir_roundtrip_flags",
        ":tensorflow",
        ":tensorflow_attributes",
        ":tensorflow_passes",
        ":tensorflow_types",
        ":tf_saved_model_passes",
        ":translate_utils",
        ":upgrade_graph",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/cc/saved_model:loader_lite",
        "//tensorflow/cc/saved_model:loader_util",
        "//tensorflow/compiler/jit:shape_inference_helpers",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/utils:transitive_fanin",
        "//tensorflow/core/platform:crash_analysis",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "parse_text_proto",
    srcs = ["utils/parse_text_proto.cc"],
    hdrs = ["utils/parse_text_proto.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:casts",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "import_utils",
    srcs = ["utils/import_utils.cc"],
    hdrs = ["utils/import_utils.h"],
    deps = [
        ":error_util",
        ":parse_text_proto",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "export_utils",
    srcs = [
        "utils/export_utils.cc",
    ],
    hdrs = [
        "utils/export_utils.h",
    ],
    deps = [
        ":convert_tensor",
        ":convert_type",
        ":location_utils",
        ":mangling_util",
        ":tensorflow",
        ":tensorflow_attributes",
        ":tensorflow_types",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:protobuf",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "location_utils",
    srcs = ["utils/location_utils.cc"],
    hdrs = ["utils/location_utils.h"],
    deps = [
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tpu_cluster_util",
    srcs = ["utils/tpu_cluster_util.cc"],
    hdrs = ["utils/tpu_cluster_util.h"],
    deps = [
        ":device_util",
        ":tpu_rewrite_device_util",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "translate_utils",
    srcs = [
        "utils/translate_utils.cc",
    ],
    hdrs = [
        "utils/translate_utils.h",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "export_tf_dialect_op",
    srcs = [
        "translate/export_tf_dialect_op.cc",
    ],
    hdrs = [
        "translate/export_tf_dialect_op.h",
    ],
    deps = [
        ":convert_type",
        ":export_utils",
        ":tensorflow",
        "//tensorflow/compiler/mlir/utils:string_container_utils",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "translate_tf_dialect_op",
    srcs = ["translate/translate_tf_dialect_op.cc"],
    deps = [
        ":export_tf_dialect_op",
        ":tensorflow",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_roundtrip_pass",
    srcs = ["translate/mlir_roundtrip_pass.cc"],
    hdrs = ["translate/mlir_roundtrip_pass.h"],
    deps = [
        ":error_util",
        ":export_graphdef",
        ":import_model",
        ":mlir_roundtrip_flags",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "mlir_roundtrip_pass_registration",
    srcs = ["translate/mlir_roundtrip_pass_registration.cc"],
    deps = [
        ":mlir_roundtrip_pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_roundtrip_flags",
    srcs = ["translate/mlir_roundtrip_flags.cc"],
    hdrs = ["translate/mlir_roundtrip_flags.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:types",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "convert_attr",
    srcs = ["utils/convert_attr.cc"],
    hdrs = ["utils/convert_attr.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":convert_tensor",
        ":convert_type",
        ":tensorflow_attributes",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "convert_type",
    srcs = [
        "utils/convert_type.cc",
    ],
    hdrs = [
        "utils/convert_type.h",
    ],
    textual_hdrs = [
        "ir/tf_types.def",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":dynamic_shape_utils",
        ":tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "convert_type_test",
    size = "small",
    srcs = ["utils/convert_type_test.cc"],
    deps = [
        ":convert_type",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "convert_tensor",
    srcs = ["utils/convert_tensor.cc"],
    hdrs = ["utils/convert_tensor.h"],
    deps = [
        ":convert_type",
        ":dynamic_shape_utils",
        ":mangling_util",
        ":tensorflow_attributes",
        ":tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/tsl/platform:float8",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "convert_tensor_test",
    size = "small",
    srcs = ["utils/convert_tensor_test.cc"],
    deps = [
        ":convert_tensor",
        ":dynamic_shape_utils",
        ":tensorflow",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/tsl/platform:float8",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "mangling_util",
    srcs = ["utils/mangling_util.cc"],
    hdrs = ["utils/mangling_util.h"],
    deps = [
        ":parse_text_proto",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/ir/importexport:mangling",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "error_util",
    srcs = ["utils/error_util.cc"],
    hdrs = ["utils/error_util.h"],
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        "//tensorflow/compiler/xla/mlir/utils:error_util",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/util:managed_stack_trace",
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "constant_fold_utils",
    srcs = [
        "transforms/constant_fold_utils.cc",
    ],
    hdrs = [
        "transforms/constant_fold_utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/c:tf_status",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/protobuf:for_core_protos_cc",
    ],
)

cc_library(
    name = "tf_dialect_passes",
    srcs = [
        "transforms/constant_fold.cc",
        "transforms/decode_attributes_hook.cc",
    ],
    hdrs = [
        "transforms/constant_fold.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":constant_fold_utils",
        ":convert_tensor",
        ":eval_util",
        ":tensorflow",
        ":tensorflow_traits",
        ":tensorflow_types",
        "//tensorflow/c:tf_status",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_dialect_lib",
    deps = [
        ":tf_dialect_passes",
        "@llvm-project//mlir:AllPassesAndDialects",
    ],
)

cc_library(
    name = "tf_graph_optimization_pass",
    srcs = ["transforms/tf_graph_optimization_pass.cc"],
    hdrs = ["transforms/tf_graph_optimization_pass.h"],
    deps = [
        ":export_graphdef",
        ":import_model",
        ":mlir_roundtrip_flags",
        ":tensorflow",
        "//tensorflow/compiler/tf2xla:functionalize_control_flow_pass_registration",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "eval_util",
    srcs = ["utils/eval_util.cc"],
    hdrs = ["utils/eval_util.h"],
    deps = [
        ":convert_tensor",
        ":export_tf_dialect_op",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "mlir_import_options",
    hdrs = ["translate/mlir_import_options.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "translate_lib",
    srcs = ["translate/tf_mlir_translate.cc"],
    hdrs = ["translate/tf_mlir_translate.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":error_util",
        ":import_model",
        ":import_utils",
        ":mangling_util",
        ":mlir_import_options",
        ":mlir_roundtrip_flags",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/cc/saved_model:loader",
        "//tensorflow/cc/saved_model:reader",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/utils:transitive_fanin",
        "//tensorflow/core/util/tensor_bundle:byteswaptensor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

cc_library(
    name = "translate_cl_options",
    srcs = [
        "translate/tf_mlir_translate_cl.cc",
    ],
    hdrs = [
        "translate/tf_mlir_translate_cl.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "translate_registration",
    srcs = [
        "translate/tf_mlir_translate_registration.cc",
    ],
    deps = [
        ":export_graphdef",
        ":mlir_roundtrip_flags",
        ":tensorflow",
        ":translate_cl_options",
        ":translate_lib",
        "//tensorflow/core:protos_all_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TranslateLib",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "error_util_test",
    srcs = ["utils/error_util_test.cc"],
    deps = [
        ":error_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla/mlir/utils:error_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

filegroup(
    name = "tensorflow_optimize_td_files",
    srcs = [
        "transforms/optimize.td",
    ],
)

gentbl_cc_library(
    name = "tensorflow_optimize_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_optimize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize.td",
    deps = [
        ":tensorflow_ops_td_files",
        "@llvm-project//mlir:ArithOpsTdFiles",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "serialize_mlir_module_utils",
    srcs = ["utils/serialize_mlir_module_utils.cc"],
    hdrs = ["utils/serialize_mlir_module_utils.h"],
    deps = [
        ":error_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

cc_library(
    name = "tf_xla_mlir_translate",
    testonly = True,  # Ensure alwayslink does not leak in the codebase.
    srcs = ["utils/tf_xla_mlir_translate.cc"],
    deps = [
        ":mlir_roundtrip_flags",
        ":serialize_mlir_module_utils",
        ":tensorflow",
        ":translate_cl_options",
        "//tensorflow/compiler/mlir/tf2xla:compile_mlir_util",
        "//tensorflow/compiler/mlir/utils:string_container_utils",
        "//tensorflow/compiler/tf2xla:xla_argument",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
        "@stablehlo//:stablehlo_ops",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_passthrough_op",
    srcs = ["ops/mlir_passthrough_op.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_local_var_op",
    srcs = ["ops/mlir_local_var_op.cc"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/core:framework",
    ],
    alwayslink = 1,
)

tf_gen_op_wrapper_py(
    name = "gen_mlir_passthrough_op_py",
    out = "gen_mlir_passthrough_op.py",
    compatible_with = [],
    deps = [":mlir_passthrough_op"],
)

# Library to get rewrite patterns lowering within TensorFlow.
#
# This is a separate library so that external passes can link only this library
# without linking any of the other tensorflow passes.
gentbl_cc_library(
    name = "lower_tf_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            ["-gen-rewriters"],
            "transforms/generated_lower_tf.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/lower_tf.td",
    deps = [
        ":rewrite_util_td_files",
        ":tensorflow_ops_td_files",
        "@llvm-project//mlir:FuncTdFiles",
    ],
)

cc_library(
    name = "lower_tf_lib",
    srcs = [
        "transforms/lower_tf.cc",
    ],
    hdrs = [
        "transforms/lower_tf.h",
    ],
    deps = [
        ":dynamic_shape_utils",
        ":lower_tf_inc_gen",
        ":rewrite_util",
        ":tensorflow",
        ":tensorflow_ops",
        ":tensorflow_types",
        "//tensorflow/core:framework",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "parallel_execute_util",
    srcs = ["utils/parallel_execute_util.cc"],
    hdrs = ["utils/parallel_execute_util.h"],
    deps = [
        ":tensorflow",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "cluster_util",
    srcs = ["utils/cluster_util.cc"],
    hdrs = ["utils/cluster_util.h"],
    deps = [
        ":tensorflow_analysis",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

tf_cc_test(
    name = "cluster_util_test",
    size = "small",
    srcs = ["utils/cluster_util_test.cc"],
    deps = [
        ":cluster_util",
        ":serialize_mlir_module_utils",
        ":tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:errors",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
    ],
)

cc_library(
    name = "tpu_rewrite_device_util",
    srcs = ["utils/tpu_rewrite_device_util.cc"],
    hdrs = ["utils/tpu_rewrite_device_util.h"],
    deps = [
        ":tensorflow",
        "//tensorflow/compiler/mlir/utils:string_container_utils",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "tpu_rewrite_device_util_test",
    size = "small",
    srcs = ["utils/tpu_rewrite_device_util_test.cc"],
    deps = [
        ":device_util",
        ":tpu_rewrite_device_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "device_util",
    srcs = ["utils/device_util.cc"],
    hdrs = ["utils/device_util.h"],
    deps = [
        ":tensorflow",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "device_util_test",
    size = "small",
    srcs = ["utils/device_util_test.cc"],
    deps = [
        ":device_util",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/ir/types:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "dump_mlir_util",
    srcs = ["utils/dump_mlir_util.cc"],
    hdrs = ["utils/dump_mlir_util.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:crash_analysis",
        "//tensorflow/core/platform:logging",
        "//tensorflow/tsl/lib/io:buffered_file",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "dump_mlir_util_test",
    size = "small",
    srcs = ["utils/dump_mlir_util_test.cc"],
    deps = [
        ":dump_mlir_util",
        ":tensorflow",
        ":tensorflow_passes",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:test",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "dump_graph",
    srcs = ["utils/dump_graph.cc"],
    hdrs = ["utils/dump_graph.h"],
    deps = [
        ":error_util",
        ":tensorflow",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core/ir/importexport:graphdef_import",
        "//tensorflow/core/platform:logging",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "dump_graph_test",
    size = "small",
    srcs = ["utils/dump_graph_test.cc"],
    tags = [
        "no_windows",  # b/208469759
    ],
    deps = [
        ":dump_graph",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:test",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_test(
    name = "bridge_logger_test",
    size = "small",
    srcs = ["utils/bridge_logger_test.cc"],
    deps = [
        ":bridge_logger",
        ":serialize_mlir_module_utils",
        ":tensorflow",
        ":tensorflow_passes",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:test",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "bridge_logger",
    srcs = ["utils/bridge_logger.cc"],
    hdrs = ["utils/bridge_logger.h"],
    deps = [
        ":dump_mlir_util",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "xla_sharding_util",
    srcs = [
        "utils/xla_sharding_util.cc",
    ],
    hdrs = [
        "utils/xla_sharding_util.h",
    ],
    deps = [
        ":tensorflow",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "attribute_utils",
    srcs = ["utils/attribute_utils.cc"],
    hdrs = ["utils/attribute_utils.h"],
    deps = [
        "//tensorflow/compiler/tf2xla:tf2xla_defs",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "shape_inference_utils",
    srcs = ["utils/shape_inference_utils.cc"],
    hdrs = ["utils/shape_inference_utils.h"],
    deps = [
        ":export_tf_dialect_op",
        ":tensorflow_types",
        "//tensorflow/core/ir:shape_inference_utils",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "dynamic_shape_utils",
    srcs = ["utils/dynamic_shape_utils.cc"],
    hdrs = ["utils/dynamic_shape_utils.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "verification_utils",
    srcs = ["utils/verification_utils.cc"],
    hdrs = ["utils/verification_utils.h"],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "verify_suitable_for_graph_export",
    srcs = ["utils/verify_suitable_for_graph_export.cc"],
    hdrs = ["utils/verify_suitable_for_graph_export.h"],
    deps = [
        ":tensorflow",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "resource_value_typed_analyzer",
    srcs = ["analysis/resource_value_typed_analyzer.cc"],
    hdrs = ["analysis/resource_value_typed_analyzer.h"],
    deps = [
        ":tensorflow",
        ":tensorflow_types",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "tf_arith_ops_folder",
    srcs = ["ir/tf_arith_ops_folder.cc"],
    hdrs = ["ir/tf_arith_ops_folder.h"],
    deps = [
        ":tensorflow_types",
        "//tensorflow/core/ir/types:Dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_ops_canonicalization_helper",
    hdrs = ["ir/tf_ops_canonicalization_helper.h"],
    deps = [
        ":attribute_utils",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_ops_device_helper",
    srcs = ["ir/tf_ops_device_helper.cc"],
    hdrs = ["ir/tf_ops_device_helper.h"],
    deps = [
        ":tensorflow_structs",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
    ],
)

cc_library(
    name = "tf_ops_layout_helper",
    srcs = ["ir/tf_ops_layout_helper.cc"],
    hdrs = ["ir/tf_ops_layout_helper.h"],
    deps = [
        ":tensorflow_op_interfaces",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_ops_tensor_helper",
    srcs = ["ir/tf_ops_tensor_helper.cc"],
    hdrs = ["ir/tf_ops_tensor_helper.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tpu_embedding_ops_registry",
    srcs = [
        "ir/tpu_embedding_ops_registry.cc",
    ],
    hdrs = [
        "ir/tpu_embedding_ops_registry.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "set_tpu_infeed_layout",
    srcs = ["transforms/set_tpu_infeed_layout.cc"],
    hdrs = ["transforms/set_tpu_infeed_layout.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/stream_executor/tpu:c_api_conversions",
        "//tensorflow/compiler/xla/stream_executor/tpu:tpu_api",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:type_to_shape",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "mlprogram_util",
    srcs = ["utils/mlprogram_util.cc"],
    hdrs = ["utils/mlprogram_util.h"],
    deps = [
        ":mlprogram",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

build_test(
    name = "tensorflow_build_test",
    targets = [
        ":tensorflow",
    ],
)
# copybara:uncomment_begin(google-only)
#
# # Generate new_tf_generated_ops.td. Without --update-existing, this file can
# # replace tf_generated_ops.td. However, right now, we only use it to determine
# # whether the latter is stale.
# dialectgen(
#     name = "tf-generated-ops",
#     output = "new_tf_generated_ops.td",
# )
#
# sh_test(
#     name = "tf_generated_ops_not_stale",
#     srcs = ["ir/tf_generated_ops_not_stale.sh"],
#     args = [
#         "$(location //tensorflow/compiler/mlir/tensorflow:ir/tf_generated_ops.td)",
#         "$(location :new_tf_generated_ops.td)",
#     ],
#     data = [
#         ":new_tf_generated_ops.td",
#         "//tensorflow/compiler/mlir/tensorflow:ir/tf_generated_ops.td",
#     ],
# )
#
# copybara:uncomment_end(google-only)
