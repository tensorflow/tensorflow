# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "get_compatible_with_cloud")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//third_party/mlir:tblgen.bzl", "gentbl")
load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_gen_op_wrapper_py")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["//third_party/mlir:subpackages"],
    packages = [
        "//learning/brain/experimental/dtensor/...",
        "//learning/brain/experimental/mlir/tflite/tfmrt/...",
        "//learning/brain/experimental/tfrt/...",
        "//learning/pathways/data_parallel/tf2xla/...",
        "//platforms/xla/sparse_core/...",
        "//tensorflow/compiler/...",
        "//tensorflow/lite/experimental/tf_runtime/...",
        "//tensorflow/python/...",
    ],
)

exports_files([
    "ir/tf_generated_ops.td",
    "ir/tf_op_base.td",
    "ir/tf_op_interfaces.td",
    "ir/tf_ops.td",
])

filegroup(
    name = "tensorflow_ops_td_files",
    srcs = [
        "ir/tf_generated_ops.td",
        "ir/tf_op_base.td",
        "ir/tf_op_interfaces.td",
        "ir/tf_ops.td",
        "ir/tfrt_ops.td",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@llvm-project//mlir:include/mlir/IR/SymbolInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/CallInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/ControlFlowInterfaces.td",
        "@llvm-project//mlir:include/mlir/Interfaces/InferTypeOpInterface.td",
        "@llvm-project//mlir:include/mlir/Interfaces/LoopLikeInterface.td",
    ],
)

gentbl(
    name = "tensorflow_op_interfaces_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-op-interface-decls",
            "ir/tf_op_interfaces.h.inc",
        ),
        (
            "-gen-op-interface-defs",
            "ir/tf_op_interfaces.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_op_interfaces.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
    ],
    test = True,
)

gentbl(
    name = "tensorflow_struct_doc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-dialect-doc",
            "g3doc/tf_ops.md",
        ),
        (
            "-gen-struct-attr-decls",
            "ir/tf_structs.h.inc",
        ),
        (
            "-gen-struct-attr-defs",
            "ir/tf_structs.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_ops.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
    ],
    test = True,
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

gentbl(
    name = "tensorflow_all_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tf_all_ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tf_all_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_ops.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "tensorflow_tfrt_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tfrt_ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tfrt_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tfrt_ops.td",
    td_srcs = [
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
    gentbl(
        name = "tensorflow_" + target["name"] + "_inc_gen",
        compatible_with = get_compatible_with_cloud(),
        tbl_outs = [
            (
                "-gen-op-decls -op-include-regex=" + target["include"],
                "ir/tf_" + target["name"] + ".h.inc",
            ),
            (
                "-gen-op-defs -op-include-regex=" + target["include"],
                "ir/tf_" + target["name"] + ".cc.inc",
            ),
        ],
        tblgen = "@llvm-project//mlir:mlir-tblgen",
        td_file = "ir/tf_ops.td",
        td_srcs = [
            ":tensorflow_ops_td_files",
        ],
    ),
] for target in tf_ops_category_list]

gentbl(
    name = "tensorflow_remaining_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-op-decls -op-exclude-regex=" + "|".join([target["include"] for target in tf_ops_category_list]),
            "ir/tf_remaining_ops.h.inc",
        ),
        (
            "-gen-op-defs -op-exclude-regex=" + "|".join([target["include"] for target in tf_ops_category_list]),
            "ir/tf_remaining_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_ops.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "tf_saved_model_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tf_saved_model.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tf_saved_model.cc.inc",
        ),
        (
            "-gen-dialect-doc",
            "g3doc/tf_saved_model.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_saved_model_ops.td",
    td_srcs = [
        "@llvm-project//mlir:include/mlir/IR/OpBase.td",
        "@llvm-project//mlir:include/mlir/Dialect/StandardOps/IR/Ops.td",
        "@llvm-project//mlir:include/mlir/IR/SymbolInterfaces.td",
    ],
    test = True,
)

gentbl(
    name = "tensorflow_executor_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tf_executor.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tf_executor.cc.inc",
        ),
        (
            "-gen-dialect-doc",
            "g3doc/tf_executor.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_executor_ops.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
        "@llvm-project//mlir:include/mlir/Dialect/StandardOps/IR/Ops.td",
        "@llvm-project//mlir:include/mlir/IR/OpBase.td",
    ],
    test = True,
)

gentbl(
    name = "tensorflow_device_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-op-decls ",
            "ir/tf_device.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tf_device.cc.inc",
        ),
        (
            "-gen-dialect-doc",
            "g3doc/tf_device.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/tf_device_ops.td",
    td_srcs = [
        "@llvm-project//mlir:SideEffectTdFiles",
        "@llvm-project//mlir:include/mlir/Dialect/StandardOps/IR/Ops.td",
        "@llvm-project//mlir:include/mlir/IR/OpBase.td",
        "@llvm-project//mlir:include/mlir/Interfaces/ControlFlowInterfaces.td",
    ],
    test = True,
)

gentbl(
    name = "tensorflow_canonicalize_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_canonicalize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/canonicalize.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "hlo_legalize_tf_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_legalize_hlo.inc"),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/legalize_hlo_patterns.td",
    td_relative_includes = [
        "../hlo/include",
    ],
    td_srcs = [
        "//tensorflow/compiler/mlir/hlo:hlo_ops_td_files",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

cc_library(
    name = "tf_legalize_hlo",
    srcs = [
        "transforms/generated_legalize_hlo.inc",
        "transforms/legalize_hlo.cc",
    ],
    deps = [
        ":lower_tf_lib",
        ":tensorflow",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:hlo_ops_base_structs",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_attributes",
    srcs = [
        "ir/tf_attributes.cc",
    ],
    hdrs = [
        "ir/tf_attributes.h",
        "ir/tf_dialect.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
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
        ":tensorflow_types",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Support",
    ],
)

[[
    cc_library(
        name = "tensorflow_" + target["name"],
        srcs = [
            "ir/tf_dialect.h",
            "ir/tf_ops.h",
            "ir/tfrt_ops.h",
            "ir/tf_remaining_ops.h",
            "ir/tf_" + target["name"] + ".cc",
            "ir/tf_" + target["name"] + ".cc.inc",
        ] + ["ir/tf_" + target["name"] + ".h" for target in tf_ops_category_list],
        hdrs = [
        ],
        textual_hdrs = [
            "ir/tf_all_ops.h.inc",
            "ir/tf_ops_helpers.inc",
            "ir/tfrt_ops.h.inc",
            "ir/tf_remaining_ops.h.inc",
        ] + ["ir/tf_" + target["name"] + ".h.inc" for target in tf_ops_category_list],
        deps = [
            ":attribute_utils",
            ":tensorflow_attributes",
            ":tensorflow_canonicalize_inc_gen",
            ":tensorflow_op_interfaces",
            ":tensorflow_op_interfaces_inc_gen",
            ":tensorflow_side_effects",
            ":tensorflow_structs",
            ":tensorflow_traits",
            ":tensorflow_types",
            "@llvm-project//llvm:Support",
            "@llvm-project//mlir:ControlFlowInterfaces",
            "@llvm-project//mlir:DerivedAttributeOpInterface",
            "@llvm-project//mlir:Dialect",
            "@llvm-project//mlir:IR",
            "@llvm-project//mlir:InferTypeOpInterface",
            "@llvm-project//mlir:LoopLikeInterface",
            "@llvm-project//mlir:Parser",
            "@llvm-project//mlir:SideEffects",
            "@llvm-project//mlir:StandardOps",
            "@llvm-project//mlir:Support",
            "//tensorflow/core:framework",
            "//tensorflow/core:lib",
        ] + [":tensorflow_" + target["name"] + "_inc_gen"],
    ),
] for target in tf_ops_category_list]

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
        "ir/tf_ops_helpers.inc",
        "ir/tf_remaining_ops.h.inc",
        "ir/tfrt_ops.h.inc",
    ] + ["ir/tf_" + target["name"] + ".h.inc" for target in tf_ops_category_list],
    deps = [
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
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
        "ir/tf_ops_helpers.inc",
        "ir/tfrt_ops.h.inc",
        "ir/tf_remaining_ops.h.inc",
    ] + ["ir/tf_" + target["name"] + ".h.inc" for target in tf_ops_category_list],
    deps = [
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
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
        ":tensorflow_all_ops_inc_gen",
        ":tensorflow_tfrt_ops_inc_gen",
        ":tensorflow_remaining_ops_inc_gen",
        ":tensorflow_attributes",
        ":tensorflow_canonicalize_inc_gen",
        ":tensorflow_op_interfaces",
        ":tensorflow_op_interfaces_inc_gen",
        ":tensorflow_side_effects",
        ":tensorflow_structs",
        ":tensorflow_traits",
        ":tensorflow_types",
        ":tensorflow_remaining_ops",
        ":tensorflow_tfrt_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ] + [":tensorflow_" + target["name"] for target in tf_ops_category_list],
)

cc_library(
    name = "tensorflow_structs",
    srcs = [
        "ir/tf_structs.cc",
    ],
    hdrs = [
        "ir/tf_structs.h",
    ],
    textual_hdrs = [
        "ir/tf_structs.h.inc",
    ],
    deps = [
        ":tensorflow_struct_doc_gen",
        "//tensorflow/core:framework",
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
    deps = ["@llvm-project//mlir:SideEffects"],
)

cc_library(
    name = "tensorflow_types",
    srcs = [
        "ir/tf_types.cc",
    ],
    hdrs = [
        "ir/tf_dialect.h",
        "ir/tf_types.h",
    ],
    textual_hdrs = [
        "ir/tf_types.def",
    ],
    deps = [
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
    ],
    hdrs = [
        "dialect_registration.h",
        "ir/tf_device.h",
        "ir/tf_dialect.h",
        "ir/tf_executor.h",
        "ir/tf_ops.h",
        "ir/tf_saved_model.h",
        "ir/tf_structs.h",
        "transforms/bridge.h",
        "transforms/einsum.h",
        "transforms/passes.h",
        "transforms/unroll_batch_matmul.h",
        "@llvm-project//mlir:include/mlir/Interfaces/CallInterfaces.h",
        "@llvm-project//mlir:include/mlir/Transforms/InliningUtils.h",
    ],
    includes = ["include"],
    deps = [
        ":error_util",
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
        ":tf_pass_inc_gen",
        ":tf_saved_model_inc_gen",
        "//tensorflow/compiler/mlir/lite:validators",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:CallOpInterfacesIncGen",
        "@llvm-project//mlir:ComplexDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    # TODO(jpienaar): Merge in the dialect registration.
    alwayslink = 1,
)

gentbl(
    name = "decompose_resource_ops_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_decompose_resource_ops.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/decompose_resource_ops.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
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
        ":tensorflow",
        ":tensorflow_types",
        "//tensorflow/core:framework",
        "@llvm-project//mlir:IR",
    ],
)

gentbl(
    name = "tf_data_optimization_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_tf_data_optimization.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tf_data_optimization.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
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
        "//tensorflow/core:framework",
        "@com_google_absl//absl/memory",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_saved_model_passes",
    srcs = [
        "transforms/deduplicate_bound_input_bindings.cc",
        "transforms/freeze_global_tensors.cc",
        "transforms/freeze_saved_model_assets.cc",
        "transforms/lift_variables_pass.cc",
        "transforms/optimize_global_tensors.cc",
        "transforms/remove_vars_in_session_initializer.cc",
    ],
    hdrs = [
        "transforms/tf_saved_model_passes.h",
    ],
    deps = [
        ":lift_variables_lib",
        ":tensorflow",
        ":tensorflow_passes",
        ":tensorflow_types",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_analysis",
    srcs = [
        "analysis/per_function_aggregate_analysis.h",
        "analysis/resource_alias_analysis.cc",
        "analysis/side_effect_analysis.cc",
    ],
    hdrs = [
        "analysis/resource_alias_analysis.h",
        "analysis/side_effect_analysis.h",
    ],
    deps = [
        ":tensorflow",
        ":tensorflow_op_interfaces",
        ":tensorflow_side_effects",
        ":tensorflow_types",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

gentbl(
    name = "tf_pass_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-pass-decls -name TensorFlow",
            "transforms/tf_passes.h.inc",
        ),
        (
            "-gen-pass-doc",
            "g3doc/includes/tf_passes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tf_passes.td",
    td_srcs = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

gentbl(
    name = "tf_savedmodel_pass_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-pass-decls -name TensorFlowSavedModel",
            "transforms/tf_savedmodel_passes.h.inc",
        ),
        (
            "-gen-pass-doc",
            "g3doc/includes/tf_savedmodel_passes.md",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/tf_savedmodel_passes.td",
    td_srcs = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "tensorflow_passes",
    srcs = [
        "transforms/annotate_parameter_replication.cc",
        "transforms/batchmatmul_to_einsum.cc",
        "transforms/bridge.cc",
        "transforms/bridge_pass.cc",
        "transforms/cluster_formation.cc",
        "transforms/cluster_ops_by_policy_pass.cc",
        "transforms/cluster_outlining.cc",
        "transforms/cluster_tf_ops_pass.cc",
        "transforms/collection_ops_util.cc",
        "transforms/constant_op_device_assignment.cc",
        "transforms/cross_host_transfer.cc",
        "transforms/decompose_resource_ops_pass.cc",
        "transforms/device_attribute_to_launch.cc",
        "transforms/device_index_selector.cc",
        "transforms/drop_while_shape_invariant.cc",
        "transforms/einsum.cc",
        "transforms/executor_island_coarsening.cc",
        "transforms/executor_tpuv1_inline_tpu_island.cc",
        "transforms/executor_tpuv1_island_coarsening.cc",
        "transforms/executor_tpuv1_outline_tpu_island.cc",
        "transforms/fold_broadcast.cc",
        "transforms/functional_control_flow_to_cfg.cc",
        "transforms/functional_control_flow_to_regions.cc",
        "transforms/fused_kernel_matcher.cc",
        "transforms/generated_canonicalize.inc",
        "transforms/generated_optimize.inc",
        "transforms/gpu_fusion.cc",
        "transforms/graph_pruning.cc",
        "transforms/guarantee_all_funcs_one_use.cc",
        "transforms/init_text_file_to_import.cc",
        "transforms/launch_to_device_attribute.cc",
        "transforms/layout_optimization.cc",
        "transforms/mark_ops_for_outside_compilation.cc",
        "transforms/materialize_mlir_passthrough_op.cc",
        "transforms/merge_control_flow.cc",
        "transforms/optimize.cc",
        "transforms/outside_compiled_to_host_launch.cc",
        "transforms/parallel_execute_to_islands.cc",
        "transforms/prepare_tpu_computation_for_tf_export.cc",
        "transforms/promote_resources_to_args.cc",
        "transforms/readonly_references_to_resources.cc",
        "transforms/region_control_flow_to_functional.cc",
        "transforms/replicate_invariant_op_hoisting.cc",
        "transforms/replicate_to_island.cc",
        "transforms/resource_device_inference.cc",
        "transforms/resource_op_lifting.cc",
        "transforms/resource_op_lifting_cleanup.cc",
        "transforms/resource_op_lifting_cleanup.h",
        "transforms/rewrite_tpu_embedding_ops.cc",
        "transforms/shape_inference.cc",
        "transforms/shape_inference_pass.cc",
        "transforms/sink_constant.cc",
        "transforms/stack_ops_decomposition.cc",
        "transforms/tensor_array_ops_decomposition.cc",
        "transforms/tensor_device_copy_conversion.cc",
        "transforms/tensor_list_ops_decomposition.cc",
        "transforms/test_resource_alias_analysis.cc",
        "transforms/test_side_effect_analysis.cc",
        "transforms/test_visitor_util.cc",
        "transforms/tf_data_optimization_pass.cc",
        "transforms/tf_device_assignment.cc",
        "transforms/tf_device_replication_pass.cc",
        "transforms/tpu_cluster_cleanup_attributes.cc",
        "transforms/tpu_cluster_formation.cc",
        "transforms/tpu_colocate_composite_resource_ops.cc",
        "transforms/tpu_compile_op_replication_pass.cc",
        "transforms/tpu_device_propagation.cc",
        "transforms/tpu_dynamic_layout_pass.cc",
        "transforms/tpu_dynamic_padding_mapper.cc",
        "transforms/tpu_extract_head_tail_outside_compilation.cc",
        "transforms/tpu_extract_outside_compilation.cc",
        "transforms/tpu_host_computation_expansion.cc",
        "transforms/tpu_identity_pruning.cc",
        "transforms/tpu_merge_variables_with_execute.cc",
        "transforms/tpu_parallel_execute_sink_resource_write.cc",
        "transforms/tpu_reorder_replicate_and_partitioned_inputs.cc",
        "transforms/tpu_resource_partitioning.cc",
        "transforms/tpu_resource_read_for_write.cc",
        "transforms/tpu_rewrite_pass.cc",
        "transforms/tpu_sharding_identification_pass.cc",
        "transforms/tpu_space_to_depth_pass.cc",
        "transforms/tpu_update_embedding_enqueue_op_inputs.cc",
        "transforms/tpu_variable_runtime_reformatting.cc",
        "transforms/verify_suitable_for_graph_export_pass.cc",
        "translate/breakup-islands.cc",
        "translate/tf_executor_to_functional.cc",
        "translate/tf_functional_to_executor.cc",
    ],
    hdrs = [
        "transforms/bridge.h",
        "transforms/collection_ops_util.h",
        "transforms/einsum.h",
        "transforms/passes.h",
        "transforms/shape_inference.h",
    ],
    includes = ["include"],
    textual_hdrs = [
        "ir/tf_ops_helpers.inc",
        "transforms/passes_detail.h",
        "transforms/savedmodel_passes_detail.h",
        "transforms/tf_passes.h.inc",
        "transforms/tf_savedmodel_passes.h.inc",
    ],
    deps = [
        ":attribute_utils",
        ":bridge_logger",
        ":convert_tensor",
        ":convert_type",
        ":decompose_resource_ops",
        ":decompose_resource_ops_inc_gen",
        ":device_util",
        ":dump_mlir_util",
        ":error_util",
        ":export_tf_dialect_op",
        ":lower_tf_lib",
        ":mangling_util",
        ":serialize_mlir_module_utils",
        ":shape_inference_utils",
        ":tensorflow",
        ":tensorflow_analysis",
        ":tensorflow_ops",
        ":tensorflow_optimize_inc_gen",
        ":tensorflow_types",
        ":tf_data_optimization",
        ":tf_pass_inc_gen",
        ":tf_savedmodel_pass_inc_gen",
        ":tpu_rewrite_device_util",
        ":translate_utils",
        ":unroll_batch_matmul_pass",
        ":verification_utils",
        ":verify_suitable_for_graph_export",
        ":visitor_util",
        ":xla_sharding_util",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/mlir/lite:validators",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf_with_tf2xla",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/client:sharding_builder",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:random",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "//tensorflow/core/protobuf/tpu:dynamic_padding_proto_cc",
        "//tensorflow/core/tpu:tpu_embedding_optimization_parameters_utils",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Rewrite",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TensorDialect",
        "@llvm-project//mlir:TransformUtils",
        "@llvm-project//mlir:Transforms",
    ],
    # TODO(jpienaar): Merge in the dialect registration.
    alwayslink = 1,
)

cc_library(
    name = "tensorflow_test_passes",
    srcs = [
        "transforms/init_text_file_to_import_test_pass.cc",
        "transforms/lift_variables_test_pass.cc",
        "transforms/lower_tf_pass.cc",
    ],
    deps = [
        ":lift_variables_lib",
        ":lower_tf_lib",
        ":tensorflow",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:threadpool_options",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "graph_optimization_pass_registration",
    srcs = ["transforms/graph_optimization_pass_registration.cc"],
    deps = [
        ":graph_optimization_pass",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass",
        "//tensorflow/compiler/mlir:mlir_graph_optimization_pass_registration",
    ],
    alwayslink = 1,
)

cc_library(
    name = "upgrade_graph",
    srcs = ["translate/upgrade_graph.cc"],
    hdrs = ["translate/upgrade_graph.h"],
    deps = [
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
        "//tensorflow/stream_executor/lib",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "convert_graphdef",
    srcs = [
        "translate/export_graphdef.cc",
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
        ":error_util",
        ":export_tf_dialect_op",
        ":export_utils",
        ":mangling_util",
        ":mlir_import_options",
        ":mlir_roundtrip_flags",
        ":tensorflow",
        ":tensorflow_attributes",
        ":tensorflow_types",
        ":tf_saved_model_passes",
        ":translate_utils",
        ":upgrade_graph",
        ":verify_suitable_for_graph_export",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/cc/saved_model:constants",
        "//tensorflow/cc/saved_model:loader_lite",
        "//tensorflow/cc/saved_model:loader_util",
        "//tensorflow/compiler/jit:shape_inference_helpers",
        "//tensorflow/compiler/mlir:name_utils",
        "//tensorflow/compiler/mlir:op_or_arg_name_mapper",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/utils:transitive_fanin",
        "//tensorflow/core/platform:types",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
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
        ":mangling_util",
        ":tensorflow",
        ":tensorflow_attributes",
        ":tensorflow_types",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
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
        "//tensorflow/stream_executor/lib",
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
        "//tensorflow/compiler/mlir:string_container_utils",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_roundtrip_pass",
    srcs = ["translate/mlir_roundtrip_pass.cc"],
    hdrs = ["translate/mlir_roundtrip_pass.h"],
    deps = [
        ":convert_graphdef",
        ":error_util",
        ":mlir_roundtrip_flags",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
    ],
    alwayslink = 1,
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
    name = "mlir_import_pass_registration",
    srcs = ["translate/mlir_import_pass_registration.cc"],
    deps = [
        ":mlir_roundtrip_pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_roundtrip_flags",
    srcs = ["translate/mlir_roundtrip_flags.cc"],
    hdrs = ["translate/mlir_roundtrip_flags.h"],
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
        "//tensorflow/stream_executor/lib",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "convert_type",
    srcs = ["utils/convert_type.cc"],
    hdrs = ["utils/convert_type.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":tensorflow_types",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
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
        "//tensorflow/stream_executor/lib",
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
        ":mangling_util",
        ":tensorflow_attributes",
        ":tensorflow_types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
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
        ":tensorflow",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/stream_executor/lib",
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
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "error_util",
    srcs = ["utils/error_util.cc"],
    hdrs = ["utils/error_util.h"],
    deps = [
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "decode_constant_pass",
    srcs = [
        "transforms/decode_constant.cc",
    ],
    hdrs = [
        "transforms/decode_constant.h",
    ],
    deps = [
        ":convert_tensor",
        ":tensorflow",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
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
    deps = [
        ":convert_tensor",
        ":decode_constant_pass",
        ":eval_util",
        ":tensorflow",
        ":tensorflow_traits",
        ":tensorflow_types",
        "//tensorflow/c:tf_status",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_dialect_lib",
    deps = [
        ":tf_dialect_passes",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
    ],
)

cc_library(
    name = "tf_graph_optimization_pass",
    srcs = ["transforms/tf_graph_optimization_pass.cc"],
    hdrs = ["transforms/tf_graph_optimization_pass.h"],
    deps = [
        ":convert_graphdef",
        ":mlir_roundtrip_flags",
        ":tensorflow",
        "//tensorflow/compiler/tf2xla:functionalize_control_flow_pass_registration",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "eval_util",
    srcs = ["utils/eval_util.cc"],
    hdrs = ["utils/eval_util.h"],
    deps = [
        ":convert_tensor",
        ":convert_type",
        ":export_tf_dialect_op",
        ":mangling_util",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/c/eager:c_api_internal",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "mlir_import_options",
    hdrs = ["translate/mlir_import_options.h"],
)

cc_library(
    name = "translate_lib",
    srcs = ["translate/tf_mlir_translate.cc"],
    hdrs = ["translate/tf_mlir_translate.h"],
    deps = [
        ":convert_graphdef",
        ":error_util",
        ":import_utils",
        ":mangling_util",
        ":mlir_import_options",
        ":mlir_roundtrip_flags",
        "//tensorflow/cc/saved_model:bundle_v2",
        "//tensorflow/cc/saved_model:reader",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/utils:transitive_fanin",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
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
        ":convert_graphdef",
        ":mlir_roundtrip_flags",
        ":tensorflow",
        ":translate_cl_options",
        ":translate_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Translation",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "error_util_test",
    srcs = ["utils/error_util_test.cc"],
    deps = [
        ":error_util",
        "//tensorflow/compiler/xla:test",
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

gentbl(
    name = "tensorflow_optimize_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_optimize.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/optimize.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
    ],
)

COMPILE_MLIR_UTIL_DEPS = [
    ":bridge_logger",
    ":convert_graphdef",
    ":convert_tensor",
    ":convert_type",
    ":dump_mlir_util",
    ":error_util",
    ":mlir_roundtrip_flags",
    ":serialize_mlir_module_utils",
    ":tensorflow",
    ":tensorflow_types",
    ":tensorflow_passes",
    ":translate_utils",
    "@com_google_absl//absl/types:optional",
    "@com_google_absl//absl/types:variant",
    "@llvm-project//llvm:Support",
    "@llvm-project//mlir:IR",
    "@llvm-project//mlir:Pass",
    "@llvm-project//mlir:Shape",
    "@llvm-project//mlir:StandardOps",
    "@llvm-project//mlir:TransformUtils",
    "@llvm-project//mlir:Transforms",
    "//tensorflow/compiler/mlir/hlo:hlo",
    "//tensorflow/compiler/mlir/hlo:hlo_dialect_registration",
    "//tensorflow/compiler/mlir/hlo:sink_constants_to_control_flow",
    "//tensorflow/compiler/mlir/xla:mlir_hlo_to_hlo",
    "//tensorflow/compiler/mlir/xla:xla_passes",
    "//tensorflow/compiler/mlir/xla:type_to_shape",
    "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
    "//tensorflow/compiler/mlir/xla:xla_legalize_tf_with_tf2xla",
    "//tensorflow/compiler/tf2xla:common",
    "//tensorflow/compiler/tf2xla:xla_helpers",
    "//tensorflow/compiler/tf2xla:xla_argument",
    "//tensorflow/compiler/xla:shape_util",
    "//tensorflow/compiler/xla:xla_data_proto_cc",
    "//tensorflow/compiler/xla/client:xla_computation",
    "//tensorflow/compiler/xla/service:hlo",
    "//tensorflow/core:framework",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core/common_runtime:core_cpu_internal",
    "//tensorflow/core/platform:logging",
    "//tensorflow/core/tpu:tpu_defs",
    "//tensorflow/stream_executor/lib",
]

# Prefer to link 'compile_mlir_util' library that also links necessary
# TensorFlow passes to the pipeline. This library without tf passes is useful
# if the constant folding is not required on the TensorFlow dialect. For
# example, this is used in XLA ondemand compilation which compiles a single op
# at a time and doesn't require constant folding. Doing so helps avoid a
# circular dependency between c_api and tf passes.
# TODO(hinsu): Split out the constant folding hook and only exclude that in
# this target.
cc_library(
    name = "compile_mlir_util_no_tf_dialect_passes",
    srcs = ["utils/compile_mlir_util.cc"],
    hdrs = ["utils/compile_mlir_util.h"],
    deps = COMPILE_MLIR_UTIL_DEPS,
)

cc_library(
    name = "compile_mlir_util",
    hdrs = ["utils/compile_mlir_util.h"],
    deps = COMPILE_MLIR_UTIL_DEPS + [
        "compile_mlir_util_no_tf_dialect_passes",
        ":tf_dialect_passes",
    ],
)

cc_library(
    name = "compile_mlir_util_pass",
    srcs = ["utils/compile_mlir_util_pass.cc"],
    deps = [
        ":compile_mlir_util",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
    ],
)

cc_library(
    name = "tf_xla_mlir_translate",
    srcs = ["utils/tf_xla_mlir_translate.cc"],
    deps = [
        ":compile_mlir_util",
        ":mlir_roundtrip_flags",
        ":serialize_mlir_module_utils",
        ":tensorflow",
        ":translate_cl_options",
        "//tensorflow/compiler/mlir:string_container_utils",
        "//tensorflow/compiler/mlir/xla:translate_cl_options",
        "//tensorflow/compiler/mlir/xla:type_to_shape",
        "//tensorflow/compiler/tf2xla:xla_argument",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Translation",
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
gentbl(
    name = "lower_tf_inc_gen",
    compatible_with = get_compatible_with_cloud(),
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_lower_tf.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/lower_tf.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
        "@llvm-project//mlir:StdOpsTdFiles",
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
        ":lower_tf_inc_gen",
        ":tensorflow",
        ":tensorflow_ops",
        ":tensorflow_types",
        "//tensorflow/core:framework",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tpu_rewrite_device_util",
    srcs = ["utils/tpu_rewrite_device_util.cc"],
    hdrs = ["utils/tpu_rewrite_device_util.h"],
    deps = [
        ":tensorflow",
        "//tensorflow/compiler/mlir:string_container_utils",
        "//tensorflow/compiler/xla:array4d",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/service:computation_placer",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "//tensorflow/stream_executor/lib",
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
        "@llvm-project//llvm:Support",
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
        "//tensorflow/core/platform:logging",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:test",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "dump_graph",
    srcs = ["utils/dump_graph.cc"],
    hdrs = ["utils/dump_graph.h"],
    deps = [
        ":convert_graphdef",
        ":error_util",
        ":tensorflow",
        ":tensorflow_passes",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:logging",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:AllPassesAndDialectsNoRegistration",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

tf_cc_test(
    name = "dump_graph_test",
    size = "small",
    srcs = ["utils/dump_graph_test.cc"],
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
    ],
)

cc_library(
    name = "visitor_util",
    srcs = [
        "utils/visitor_util.cc",
    ],
    hdrs = [
        "utils/visitor_util.h",
    ],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
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
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "attribute_utils",
    hdrs = ["utils/attribute_utils.h"],
    deps = [
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "shape_inference_utils",
    srcs = ["utils/shape_inference_utils.cc"],
    hdrs = ["utils/shape_inference_utils.h"],
    deps = [
        ":convert_tensor",
        ":convert_type",
        ":export_tf_dialect_op",
        ":export_utils",
        ":tensorflow",
        ":tensorflow_attributes",
        ":tensorflow_types",
        "//tensorflow/compiler/mlir:array_container_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:types",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:Support",
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
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
