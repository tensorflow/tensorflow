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
        "//babelfish/device/...",
        "//learning/brain/experimental/mlir/...",
        "//learning/brain/experimental/swift_mlir/...",
        "//learning/brain/google/xla/kernels/...",
        "//learning/brain/swift/swift_mlir/...",
        "//tensorflow/compiler/mlir/...",
        "//tensorflow/compiler/tf2xla/...",
        "//tensorflow/compiler/xla/...",
        "//third_party/iree/...",
        "//third_party/mlir_edge/...",
    ],
)

exports_files(["ir/hlo_ops.td"])

filegroup(
    name = "hlo_ops_td_files",
    srcs = [
        "ir/hlo_ops.td",
        "ir/hlo_ops_base.td",
        "ir/hlo_utils.td",
        "ir/lhlo_ops.td",
        "@local_config_mlir//:OpBaseTdFiles",
    ],
)

gentbl(
    name = "hlo_ops_inc_gen",
    tbl_outs = [
        ("-gen-op-decls", "ir/hlo_ops.h.inc"),
        ("-gen-op-defs", "ir/hlo_ops.cc.inc"),
        ("-gen-struct-attr-decls", "ir/hlo_structs.h.inc"),
        ("-gen-struct-attr-defs", "ir/hlo_structs.cc.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ir/hlo_ops.td",
    td_includes = ["ir/hlo_utils.td"],
    td_srcs = [":hlo_ops_td_files"],
)

gentbl(
    name = "hlo_ops_base_inc_gen",
    tbl_outs = [
        ("-gen-op-decls", "ir/hlo_ops_base.h.inc"),
        ("-gen-op-defs", "ir/hlo_ops_base.cc.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ir/hlo_ops_base.td",
    td_srcs = [":hlo_ops_td_files"],
)

gentbl(
    name = "lhlo_ops_inc_gen",
    tbl_outs = [
        ("-gen-op-decls", "ir/lhlo_ops.h.inc"),
        ("-gen-op-defs", "ir/lhlo_ops.cc.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "ir/lhlo_ops.td",
    td_srcs = [":hlo_ops_td_files"],
)

gentbl(
    name = "xla_legalize_tf_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_legalize_tf.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/legalize_tf_patterns.td",
    td_srcs = [
        ":hlo_ops_td_files",
        "@llvm//:support",
        "@local_config_mlir//:StdOpsTdFiles",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_td_files",
    ],
)

gentbl(
    name = "xla_canonicalize_inc_gen",
    tbl_outs = [
        (
            "-gen-rewriters",
            "transforms/generated_canonicalize.inc",
        ),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/canonicalize.td",
    td_srcs = [
        ":hlo_ops_td_files",
    ],
)

cc_library(
    name = "xla_legalize_tf",
    srcs = [
        "transforms/generated_legalize_tf.inc",
        "transforms/legalize_tf.cc",
        "transforms/legalize_tf_control_flow.cc",
    ],
    deps = [
        ":convert_op_folder",
        ":hlo",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:lower_tf_lib",
        "//tensorflow/compiler/xla/client:padding",
        "//tensorflow/core:framework",
        "//tensorflow/core/kernels:conv_grad_shape_utils",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_legalize_to_affine",
    srcs = ["transforms/lhlo_legalize_to_affine.cc"],
    hdrs = ["transforms/map_lhlo_to_scalar_op.h"],
    deps = [
        ":lhlo",
        "//tensorflow/compiler/xla:status",
        "@com_google_absl//absl/memory",
        "@llvm//:support",
        "@local_config_mlir//:AffineOps",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_legalize_to_linalg",
    srcs = ["transforms/lhlo_legalize_to_linalg.cc"],
    hdrs = ["transforms/map_lhlo_to_scalar_op.h"],
    deps = [
        ":lhlo",
        "@com_google_absl//absl/memory",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Linalg",
        "@local_config_mlir//:LinalgDialectRegistration",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_legalize_to_gpu",
    srcs = ["transforms/lhlo_legalize_to_gpu.cc"],
    hdrs = ["transforms/map_lhlo_to_scalar_op.h"],
    deps = [
        ":lhlo",
        "@com_google_absl//absl/memory",
        "@llvm//:support",
        "@local_config_mlir//:GPUDialect",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Linalg",
        "@local_config_mlir//:LoopOps",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo_fuse_linalg",
    srcs = ["transforms/lhlo_fuse_linalg.cc"],
    deps = [
        ":lhlo",
        "@com_google_absl//absl/memory",
        "@local_config_mlir//:Linalg",
        "@local_config_mlir//:LinalgDialectRegistration",
        "@local_config_mlir//:Pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hlo_legalize_to_lhlo",
    srcs = ["transforms/hlo_legalize_to_lhlo.cc"],
    deps = [
        ":hlo",
        ":lhlo",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Transforms",
    ],
    alwayslink = 1,
)

gentbl(
    name = "xla_legalize_to_standard_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_legalize_to_standard.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/legalize_to_standard_patterns.td",
    td_srcs = [
        ":hlo_ops_td_files",
        "@local_config_mlir//:StdOpsTdFiles",
    ],
)

cc_library(
    name = "xla_legalize_control_flow",
    srcs = [
        "transforms/legalize_control_flow.cc",
    ],
    deps = [
        ":hlo",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_legalize_to_standard",
    srcs = ["transforms/legalize_to_standard.cc"],
    deps = [
        ":hlo",
        ":xla_legalize_to_standard_inc_gen",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
    ],
    alwayslink = 1,
)

gentbl(
    name = "xla_lower_complex_inc_gen",
    tbl_outs = [
        ("-gen-rewriters", "transforms/generated_lower_complex.inc"),
    ],
    tblgen = "@local_config_mlir//:mlir-tblgen",
    td_file = "transforms/lower_complex_patterns.td",
    td_srcs = [
        ":hlo_ops_td_files",
        "@llvm//:support",
        "@local_config_mlir//:StdOpsTdFiles",
    ],
)

cc_library(
    name = "xla_lower",
    srcs = [
        "transforms/generated_lower_complex.inc",
        "transforms/lower_complex.cc",
        "transforms/lower_general_dot.cc",
    ],
    deps = [
        ":hlo",
        ":xla_dialect_registration",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:Transforms",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hlo",
    srcs = [
        "ir/hlo_ops.cc",
        "ir/hlo_ops.cc.inc",
        "ir/hlo_ops.h.inc",
        "ir/hlo_utils.cc",
    ],
    hdrs = [
        "ir/hlo_ops.h",
        "ir/hlo_utils.h",
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    includes = ["include"],
    deps = [
        ":convert_op_folder",
        ":hlo_ops_base_inc_gen",
        ":hlo_ops_inc_gen",
        ":xla_canonicalize_inc_gen",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "lhlo",
    srcs = [
        "ir/lhlo_ops.cc",
        "ir/lhlo_ops.cc.inc",
        "ir/lhlo_ops.h.inc",
    ],
    hdrs = [
        "ir/lhlo_ops.h",
        "transforms/passes.h",
        "transforms/rewriters.h",
    ],
    includes = ["include"],
    deps = [
        ":hlo_ops_base_inc_gen",
        ":lhlo_ops_inc_gen",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TransformUtils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hlo_utils",
    srcs = ["hlo_utils.cc"],
    hdrs = ["hlo_utils.h"],
    includes = ["include"],
    deps = [
        ":convert_op_folder",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla/service:hlo",
        "@local_config_mlir//:IR",
    ],
    alwayslink = 1,
)

# Library with XLA dialect static initialization.
cc_library(
    name = "xla_dialect_registration",
    srcs = ["ir/dialect_registration.cc"],
    deps = [
        ":hlo",
        ":lhlo",
        "@local_config_mlir//:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "type_to_shape",
    srcs = ["type_to_shape.cc"],
    hdrs = ["type_to_shape.h"],
    deps = [
        ":hlo",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:types",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Support",
    ],
)

tf_cc_test(
    name = "type_to_shape_test",
    srcs = ["type_to_shape_test.cc"],
    deps = [
        ":type_to_shape",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:test",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test_main",
        "@local_config_mlir//:IR",
    ],
)

cc_library(
    name = "mlir_hlo_to_hlo",
    srcs = [
        "mlir_hlo_to_hlo.cc",
        "operator_writers.inc",
    ],
    hdrs = ["mlir_hlo_to_hlo.h"],
    deps = [
        ":hlo",
        ":type_to_shape",
        ":xla_dialect_registration",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/xla:comparison_util",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/client:xla_builder",
        "//tensorflow/compiler/xla/client/lib:matrix",
        "//tensorflow/compiler/xla/client/lib:slicing",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/stream_executor/lib",
        "@llvm//:support",
        "@local_config_mlir//:Analysis",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Pass",
        "@local_config_mlir//:StandardOps",
        "@local_config_mlir//:TransformUtils",
        "@local_config_mlir//:Transforms",
    ],
)

cc_library(
    name = "hlo_to_mlir_hlo",
    srcs = ["hlo_to_mlir_hlo.cc"],
    hdrs = ["hlo_to_mlir_hlo.h"],
    deps = [
        ":hlo_module_importer",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/core:lib",
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
        ":hlo",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/xla:protobuf_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:status_macros",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:StandardOps",
    ],
)

cc_library(
    name = "xla_mlir_translate",
    srcs = ["xla_mlir_translate.cc"],
    hdrs = ["xla_mlir_translate.h"],
    deps = [
        ":hlo_to_mlir_hlo",
        ":mlir_hlo_to_hlo",
        "//tensorflow/compiler/xla:debug_options_flags",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/service:hlo_parser",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_protobuf//:protobuf_headers",
        "@llvm//:support",
        "@local_config_mlir//:IR",
        "@local_config_mlir//:Translation",
    ],
    alwayslink = 1,
)

tf_native_cc_binary(
    name = "operator_writer_gen",
    srcs = ["operator_writer_gen.cc"],
    deps = [
        "@llvm//:support",
        "@llvm//:tablegen",
        "@local_config_mlir//:Support",
        "@local_config_mlir//:TableGen",
    ],
)

genrule(
    name = "operator_writer_inc",
    srcs = [
        "@local_config_mlir//:include/mlir/IR/OpBase.td",
        ":ir/hlo_ops.td",
        ":ir/hlo_ops_base.td",
        ":ir/hlo_utils.td",
    ],
    outs = ["operator_writers.inc"],
    cmd = ("$(location :operator_writer_gen) " +
           "-I external/local_config_mlir/include " +
           "$(location //tensorflow/compiler/mlir/xla:ir/hlo_ops.td) " +
           " -o $@"),
    tools = [":operator_writer_gen"],
)

cc_library(
    name = "convert_op_folder",
    srcs = ["convert_op_folder.cc"],
    hdrs = ["convert_op_folder.h"],
    deps = [
        "@local_config_mlir//:IR",
    ],
)
