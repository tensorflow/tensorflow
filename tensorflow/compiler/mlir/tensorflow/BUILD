# buildifier: disable=out-of-order-load

load("//tensorflow:strict.default.bzl", "py_strict_library")

# copybara:uncomment_begin(google-only)
# load("//learning/brain/experimental/mlir/tensorflow/dialectgen:dialectgen.bzl", "dialectgen")
#
# copybara:uncomment_end(google-only)
load("@bazel_skylib//rules:build_test.bzl", "build_test")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

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
    ],
    compatible_with = get_compatible_with_portable(),
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
    compatible_with = get_compatible_with_portable(),
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
    compatible_with = get_compatible_with_portable(),
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
    compatible_with = get_compatible_with_portable(),
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
        compatible_with = get_compatible_with_portable(),
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
    compatible_with = get_compatible_with_portable(),
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
    compatible_with = get_compatible_with_portable(),
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
    compatible_with = get_compatible_with_portable(),
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
    compatible_with = get_compatible_with_portable(),
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
               "ir/tf_remaining_ops.h",
           ] + ["ir/tf_" + target["name"] + ".cc" for target in tf_ops_category_list] +
           ["ir/tf_" + target["name"] + ".cc.inc" for target in tf_ops_category_list] +
           ["ir/tf_" + target["name"] + ".h" for target in tf_ops_category_list],
    hdrs = [
    ],
    textual_hdrs = [
        "ir/tf_types.def",
        "ir/tf_all_ops.h.inc",
        "ir/tf_remaining_ops.h.inc",
    ] + ["ir/tf_" + target["name"] + ".h.inc" for target in tf_ops_category_list],
    deps = [
        ":attribute_utils",
        ":convert_type",
        ":dynamic_shape_utils",
        ":side_effect_analysis_util",
        ":tensorflow_all_ops_inc_gen",
        ":tensorflow_attributes",
        ":tensorflow_op_interfaces",
        ":tensorflow_op_interfaces_inc_gen",
        ":tensorflow_remaining_ops_inc_gen",
        ":tensorflow_side_effects",
        ":tensorflow_structs",
        ":tensorflow_traits",
        ":tensorflow_types",
        ":tf_arith_ops_folder",
        ":tf_ops_canonicalization_helper",
        ":tf_ops_device_helper",
        ":tf_ops_layout_helper",
        ":tf_ops_tensor_helper",
        "//tensorflow/compiler/mlir/tensorflow/ir/host_runtime:tensorflow_tfrt_ops",
        "//tensorflow/compiler/mlir/tensorflow/ir/host_runtime:tensorflow_tfrt_ops_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow/transforms:rewrite_util",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_canonicalize_inc_gen",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
        "@llvm-project//mlir:LoopLikeInterface",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ] + [":tensorflow_" + target["name"] + "_inc_gen" for target in tf_ops_category_list],
)

cc_library(
    name = "tensorflow_remaining_ops",
    srcs = [
        "ir/tf_dialect.h",
        "ir/tf_ops.h",
        "ir/tf_remaining_ops.cc",
        "ir/tf_remaining_ops.h",
    ] + ["ir/tf_" + target["name"] + ".h" for target in tf_ops_category_list],
    hdrs = [
    ],
    textual_hdrs = [
        "ir/tf_all_ops.h.inc",
        "ir/tf_remaining_ops.h.inc",
    ] + ["ir/tf_" + target["name"] + ".h.inc" for target in tf_ops_category_list],
    deps = [
        ":attribute_utils",
        ":serialize_mlir_module_utils",
        ":side_effect_analysis_util",
        ":tensorflow_attributes",
        ":tensorflow_op_interfaces",
        ":tensorflow_op_interfaces_inc_gen",
        ":tensorflow_remaining_ops_inc_gen",
        ":tensorflow_side_effects",
        ":tensorflow_structs",
        ":tensorflow_traits",
        ":tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow/ir/host_runtime:tensorflow_tfrt_ops",
        "//tensorflow/compiler/mlir/tensorflow/ir/host_runtime:tensorflow_tfrt_ops_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow/transforms:rewrite_util",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_canonicalize_inc_gen",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
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
    ] + ["ir/tf_" + target["name"] + ".h" for target in tf_ops_category_list],
    deps = [
        ":side_effect_analysis_util",
        ":tensorflow_all_ops_inc_gen",
        ":tensorflow_attributes",
        ":tensorflow_op_interfaces",
        ":tensorflow_op_interfaces_inc_gen",
        ":tensorflow_ops_sharded",
        ":tensorflow_remaining_ops",
        ":tensorflow_remaining_ops_inc_gen",
        ":tensorflow_side_effects",
        ":tensorflow_structs",
        ":tensorflow_traits",
        ":tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow/ir/host_runtime:tensorflow_tfrt_ops",
        "//tensorflow/compiler/mlir/tensorflow/ir/host_runtime:tensorflow_tfrt_ops_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow/transforms:rewrite_util",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_canonicalize_inc_gen",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:inline_function_utils",
        "//tensorflow/core/common_runtime:lower_function_call_inline_policy",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:CallOpInterfaces",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
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
    ],
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensorflow_all_ops_inc_gen",
        ":tensorflow_attributes",
        ":tensorflow_device_ops_inc_gen",
        ":tensorflow_executor_inc_gen",
        ":tensorflow_op_interfaces",
        ":tensorflow_ops",
        ":tensorflow_side_effects",
        ":tensorflow_structs",
        ":tensorflow_traits",
        ":tensorflow_types",
        ":tf_saved_model_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow/ir/host_runtime:tensorflow_tfrt_ops",
        "//tensorflow/compiler/mlir/tensorflow/ir/host_runtime:tensorflow_tfrt_ops_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_canonicalize_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_device_pass_inc_gen",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/ir:Dialect",
        "//tensorflow/core/ir/types:Dialect",
        "//tensorflow/core/platform:logging",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:BytecodeOpInterface",
        "@llvm-project//mlir:CallOpInterfacesIncGen",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowInterfaces",
        "@llvm-project//mlir:DerivedAttributeOpInterface",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:InferTypeOpInterface",
        "@llvm-project//mlir:InliningUtils",
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
    name = "string_util",
    srcs = ["utils/string_util.cc"],
    hdrs = ["utils/string_util.h"],
    deps = [
        "@com_google_absl//absl/strings",
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
        "@llvm-project//mlir:Support",
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
        "analysis/tf_dataflow.h",
    ],
    deps = [
        ":tensorflow",
        ":tensorflow_op_interfaces",
        ":tensorflow_side_effects",
        ":tensorflow_traits",
        ":tensorflow_types",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "xla_call_module_attrs",
    srcs = [],
    hdrs = ["utils/xla_call_module_attrs.h"],
    deps = ["@llvm-project//llvm:Support"],
)

cc_library(
    name = "stablehlo_custom_call_utils",
    srcs = ["utils/stablehlo_custom_call.cc"],
    hdrs = ["utils/stablehlo_custom_call.h"],
    deps = [
        ":xla_call_module_attrs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
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
        ":attribute_utils",
        ":convert_tensor",
        ":convert_type",
        ":location_utils",
        ":mangling_util",
        ":tensorflow",
        ":tensorflow_attributes",
        ":tensorflow_types",
        "//tensorflow/compiler/mlir/tensorflow:xla_sharding_util",
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
        "@local_xla//xla:status_macros",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:sharding_builder",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/service:hlo_parser",
    ],
)

cc_library(
    name = "location_utils",
    srcs = ["utils/location_utils.cc"],
    hdrs = ["utils/location_utils.h"],
    deps = [
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
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
        "@llvm-project//mlir:IR",
        "@local_tsl//tsl/platform:statusor",
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
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@local_xla//xla:test",
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
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:ml_dtypes",
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
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:ml_dtypes",
        "@local_xla//xla:test",
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
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/util:managed_stack_trace",
        "@com_google_absl//absl/status",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_xla//xla/mlir/utils:error_util",
    ],
)

cc_library(
    name = "tf_dialect_lib",
    deps = [
        "//tensorflow/compiler/mlir/tensorflow/transforms:tf_dialect_passes",
        "@llvm-project//mlir:AllPassesAndDialects",
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

tf_cc_test(
    name = "error_util_test",
    srcs = ["utils/error_util_test.cc"],
    deps = [
        ":error_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@local_xla//xla:test",
        "@local_xla//xla/mlir/utils:error_util",
    ],
)

cc_library(
    name = "serialize_mlir_module_utils",
    srcs = ["utils/serialize_mlir_module_utils.cc"],
    hdrs = ["utils/serialize_mlir_module_utils.h"],
    deps = [
        ":error_util",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@local_xla//xla:status_macros",
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
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:AsmParser",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncExtensions",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:QuantOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TranslateLib",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/hlo/translate/mhlo_to_hlo:type_to_shape",
        "@local_xla//xla/service:hlo_module_config",
        "@local_xla//xla/service:hlo_proto_cc",
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
    extra_py_deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    deps = [":mlir_passthrough_op"],
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
        "@com_google_googletest//:gtest",
        "@llvm-project//mlir:FuncDialect",
        "@local_tsl//tsl/platform:statusor",
    ],
)

cc_library(
    name = "tpu_rewrite_device_util",
    srcs = ["utils/tpu_rewrite_device_util.cc"],
    hdrs = ["utils/tpu_rewrite_device_util.h"],
    deps = [
        ":device_util",
        ":tensorflow",
        ":tensorflow_types",
        "//tensorflow/compiler/jit:flags_headers",
        "//tensorflow/compiler/mlir/utils:string_container_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_xla//xla:array4d",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/service:computation_placer",
    ],
)

tf_cc_test(
    name = "tpu_rewrite_device_util_test",
    size = "small",
    srcs = ["utils/tpu_rewrite_device_util_test.cc"],
    deps = [
        ":device_util",
        ":serialize_mlir_module_utils",
        ":tensorflow",
        ":tpu_rewrite_device_util",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:status_matchers",
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
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@local_xla//xla/tsl/lib/io:buffered_file",
    ],
)

tf_cc_test(
    name = "dump_mlir_util_test",
    size = "small",
    srcs = ["utils/dump_mlir_util_test.cc"],
    deps = [
        ":dump_mlir_util",
        ":tensorflow",
        "//tensorflow/compiler/mlir/tensorflow/transforms:bridge",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:test",
        "@com_google_googletest//:gtest",
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
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:test",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
    ],
)

tf_cc_test(
    name = "data_dumper_logger_config_test",
    size = "small",
    srcs = ["utils/data_dumper_logger_config_test.cc"],
    deps = [
        ":bridge_logger",
        ":serialize_mlir_module_utils",
        ":tensorflow",
        "//tensorflow/compiler/mlir/tensorflow/transforms:tensorflow_passes",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:test",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "bridge_logger",
    srcs = [
        "utils/bridge_logger.cc",
        "utils/data_dumper_logger_config.cc",
    ],
    hdrs = [
        "utils/bridge_logger.h",
        "utils/data_dumper_logger_config.h",
    ],
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
    name = "call_graph_util",
    srcs = [
        "utils/call_graph_util.cc",
    ],
    hdrs = [
        "utils/call_graph_util.h",
    ],
    deps = [
        ":tensorflow",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

tf_cc_test(
    name = "call_graph_util_test",
    size = "small",
    srcs = ["utils/call_graph_util_test.cc"],
    deps = [
        ":attribute_utils",
        ":call_graph_util",
        ":tensorflow",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
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
        "//tensorflow/core:framework",
        "//tensorflow/core/protobuf/tpu:compile_metadata_proto_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/service:hlo_parser",
        "@local_xla//xla/tsl/lib/math:math_util",
    ],
)

tf_cc_test(
    name = "utils/xla_sharding_util_test",
    srcs = ["utils/xla_sharding_util_test.cc"],
    deps = [
        ":serialize_mlir_module_utils",
        ":tensorflow",
        ":xla_sharding_util",
        "//tensorflow/core:test",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_tsl//tsl/platform:statusor",
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
        "@llvm-project//mlir:Support",
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
        "@llvm-project//mlir:Support",
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
        "@llvm-project//mlir:Support",
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
        "@llvm-project//mlir:Support",
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
    name = "mlprogram_util",
    srcs = ["utils/mlprogram_util.cc"],
    hdrs = ["utils/mlprogram_util.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow/transforms:mlprogram",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "visitor",
    srcs = ["utils/visitor.cc"],
    hdrs = ["utils/visitor.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "xla_rewrite_util",
    srcs = ["utils/xla_rewrite_util.cc"],
    hdrs = ["utils/xla_rewrite_util.h"],
    deps = [
        ":device_util",
        ":tensorflow",
        ":tensorflow_types",
        "//tensorflow/compiler/mlir/utils:string_container_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf/tpu:topology_proto_cc",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@local_xla//xla:array4d",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/service:computation_placer",
    ],
)

tf_cc_test(
    name = "xla_rewrite_util_test",
    size = "small",
    srcs = ["utils/xla_rewrite_util_test.cc"],
    deps = [
        ":device_util",
        ":serialize_mlir_module_utils",
        ":tensorflow",
        ":tpu_rewrite_device_util",
        ":xla_rewrite_util",
        "//tensorflow/compiler/jit:flags",
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
    name = "side_effect_analysis_util",
    srcs = [
        "utils/side_effect_analysis_util.cc",
    ],
    hdrs = [
        "utils/side_effect_analysis_util.h",
    ],
    deps = [
        "tensorflow_side_effects",
        "tensorflow_types",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
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

# Required as we created the transforms subpackage and need to update
# these BUILD targets in a follow up.
aliased_targets = [
    "import_model",
    "export_tf_dialect_op",
    "mlir_roundtrip_flags",
    "mlir_import_options",
    "translate_lib",
    "translate_cl_options",
    "translate_registration",
    "split_into_island_per_op_pass",
    "upgrade_graph",
]

[
    alias(
        name = target,
        actual = "//tensorflow/compiler/mlir/tensorflow/translate:%s" % target,
        visibility = ["//visibility:public"],
    )
    for target in aliased_targets
]
