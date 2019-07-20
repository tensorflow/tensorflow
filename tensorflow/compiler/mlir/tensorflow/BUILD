load("@local_config_mlir//:tblgen.bzl", "gentbl")
load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_native_cc_binary")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    includes = ["@local_config_mlir//:subpackages"],
    packages = [
        "//learning/brain/experimental/mlir/...",
        "//learning/brain/google/xla/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/python/...",
    ],
)

filegroup(
    name = "tensorflow_ops_td_files",
    srcs = [
        "ir/tf_generated_ops.td",
        "ir/tf_op_base.td",
        "ir/tf_ops.td",
        "@local_config_mlir//:OpBaseTdFiles",
    ],
)

gentbl(
    name = "tensorflow_ops_inc_gen",
    tbl_outs = [
        (
            "-gen-op-decls",
            "ir/tf_ops.h.inc",
        ),
        (
            "-gen-op-defs",
            "ir/tf_ops.cc.inc",
        ),
        (
            "-gen-op-doc",
            "g3doc/tf_ops.md",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ir/tf_ops.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "tensorflow_executor_inc_gen",
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
            "-gen-op-doc",
            "g3doc/tf_executor.md",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ir/tf_executor_ops.td",
    td_srcs = [
        "@local_config_mlir//:include/mlir/IR/OpBase.td",
        "@local_config_mlir//:include/mlir/StandardOps/Ops.td",
    ],
)

gentbl(
    name = "tensorflow_canonicalize_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_canonicalize.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/canonicalize.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
    ],
)

cc_library(
    name = "tensorflow",
    srcs = [
        "ir/control_flow_ops.cc",
        "ir/tf_executor.cc",
        "ir/tf_executor.cc.inc",
        "ir/tf_executor.h.inc",
        "ir/tf_ops.cc",
        "ir/tf_ops.cc.inc",
        "ir/tf_ops.h.inc",
        "transforms/functional_control_flow_to_cfg.cc",
        "transforms/generated_canonicalize.inc",
        "transforms/generated_optimize.inc",
        "transforms/optimize.cc",
        "transforms/raise_control_flow.cc",
        "translate/control_to_executor_dialect.cc",
    ],
    hdrs = [
        "ir/control_flow_ops.h",
        "ir/tf_executor.h",
        "ir/tf_ops.h",
        "ir/tf_types.def",
        "ir/tf_types.h",
        "transforms/passes.h",
    ],
    includes = ["include"],
    deps = [
        ":tensorflow_canonicalize_inc_gen",
        ":tensorflow_executor_inc_gen",
        ":tensorflow_ops_inc_gen",
        ":tensorflow_optimize_inc_gen",
        "//tensorflow/compiler/mlir/lite:validators",
        "//tensorflow/core:lib",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:Dialect",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Parser",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TransformUtils",
    ],
    # TODO(jpienaar): Merge in the dialect registration.
    alwayslink = 1,
)

# Library with TensorFlow dialect static initialization.
cc_library(
    name = "tensorflow_dialect_registration",
    srcs = ["ir/dialect_registration.cc"],
    deps = [
        ":tensorflow",
        "@local_config_mlir//:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "convert_graphdef",
    srcs = [
        "translate/export_graphdef.cc",
        "translate/import_graphdef.cc",
    ],
    hdrs = [
        "translate/export_graphdef.h",
        "translate/import_graphdef.h",
    ],
    deps = [
        ":convert_tensor",
        ":convert_type",
        ":export_tf_dialect_op",
        ":export_utils",
        ":mangling_util",
        ":mlir_roundtrip_flags",
        ":tensorflow",
        "//tensorflow/compiler/jit:shape_inference_helpers",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:StandardDialectRegistration",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
    ],
)

cc_library(
    name = "import_utils",
    srcs = [
        "utils/import_utils.cc",
    ],
    hdrs = [
        "utils/import_utils.h",
    ],
    deps = [
        ":error_util",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
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
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:StandardDialectRegistration",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
    ],
)

cc_library(
    name = "export_tf_dialect_op",
    srcs = [
        "translate/derived_attr_populator.inc",
        "translate/export_tf_dialect_op.cc",
    ],
    hdrs = [
        "translate/export_tf_dialect_op.h",
    ],
    deps = [
        ":convert_type",
        ":export_utils",
        ":tensorflow",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm//:support",
        "@local_config_mlir//:IR",
    ],
)

cc_library(
    name = "translate_tf_dialect_op",
    srcs = ["translate/translate_tf_dialect_op.cc"],
    deps = [
        ":export_tf_dialect_op",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:Translation",
    ],
    alwayslink = 1,
)

cc_library(
    name = "mlir_roundtrip_pass",
    srcs = ["translate/mlir_roundtrip_pass.cc"],
    hdrs = ["translate/mlir_roundtrip_pass.h"],
    deps = [
        ":convert_graphdef",
        ":mlir_roundtrip_flags",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:StandardOps",
    ],
)

cc_library(
    name = "mlir_roundtrip_flags",
    srcs = ["translate/mlir_roundtrip_flags.cc"],
    hdrs = ["translate/mlir_roundtrip_flags.h"],
    deps = [
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
    ],
)

cc_library(
    name = "convert_type",
    srcs = ["utils/convert_type.cc"],
    hdrs = ["utils/convert_type.h"],
    deps = [
        ":tensorflow",
        ":tensorflow_dialect_registration",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Support",
    ],
)

cc_library(
    name = "convert_tensor",
    srcs = ["utils/convert_tensor.cc"],
    hdrs = ["utils/convert_tensor.h"],
    deps = [
        ":convert_type",
        ":mangling_util",
        ":tensorflow",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
        "@local_config_mlir//:IR",
    ],
)

cc_library(
    name = "mangling_util",
    srcs = ["utils/mangling_util.cc"],
    hdrs = ["utils/mangling_util.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "error_util",
    srcs = ["utils/error_util.cc"],
    hdrs = ["utils/error_util.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@llvm//:support",
        "@local_config_mlir//:IR",
    ],
)

cc_library(
    name = "tf_dialect_passes",
    srcs = [
        "transforms/constant_fold.cc",
        "transforms/decode_constant.cc",
        "transforms/dialect_hooks.cc",
    ],
    hdrs = [
        "transforms/constant_fold.h",
        "transforms/decode_constant.h",
    ],
    deps = [
        ":convert_tensor",
        ":eval_util",
        ":tensorflow",
        ":tf_graph_optimization_pass",
        "//tensorflow/c:tf_status",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/core:framework",
        "//tensorflow/stream_executor",
        "//tensorflow/stream_executor/lib",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_dialect_lib",
    deps = [
        ":tensorflow_dialect_registration",
        ":tf_dialect_passes",
        "@local_config_mlir//:StandardDialectRegistration",
    ],
)

cc_library(
    name = "tf_graph_optimization_pass",
    srcs = ["transforms/tf_graph_optimization_pass.cc"],
    deps = [
        ":convert_graphdef",
        ":mlir_roundtrip_flags",
        "//tensorflow/compiler/tf2xla:functionalize_control_flow_pass_registration",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
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
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Support",
    ],
)

cc_library(
    name = "translate_lib",
    srcs = [
        "translate/tf_mlir_translate.cc",
    ],
    hdrs = [
        "translate/tf_mlir_translate.h",
    ],
    deps = [
        ":convert_graphdef",
        ":error_util",
        ":import_utils",
        ":mangling_util",
        ":mlir_roundtrip_flags",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_proto_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Parser",
        "@local_config_mlir//:Pass",
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
        "@llvm//:support",
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
        ":translate_cl_options",
        ":translate_lib",
        "//tensorflow/core:protos_all_proto_cc",
        "//tensorflow/stream_executor/lib",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Translation",
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
        "@llvm//:support",
        "@local_config_mlir//:IR",
    ],
)

tf_native_cc_binary(
    name = "derived_attr_populator_gen",
    srcs = [
        "translate/derived_attr_populator_gen.cc",
    ],
    deps = [
        "@llvm//:support",
        "@llvm//:tablegen",
        "@local_config_mlir//:TableGen",
    ],
)

genrule(
    name = "derived_attr_populator_inc",
    srcs = [
        "@local_config_mlir//:include/mlir/IR/OpBase.td",
        "ir/tf_generated_ops.td",
        "ir/tf_op_base.td",
        "ir/tf_ops.td",
    ],
    outs = [
        "translate/derived_attr_populator.inc",
    ],
    cmd = ("$(location :derived_attr_populator_gen) " +
           "-I external/local_config_mlir/include " +
           "$(location //tensorflow/compiler/mlir/tensorflow:ir/tf_ops.td) " + " -o $@"),
    tools = [":derived_attr_populator_gen"],
)

filegroup(
    name = "tensorflow_optimize_td_files",
    srcs = [
        "transforms/optimize.td",
    ],
)

gentbl(
    name = "tensorflow_optimize_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_optimize.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/optimize.td",
    td_srcs = [
        ":tensorflow_ops_td_files",
        "@local_config_mlir//:StdOpsTdFiles",
    ],
)
