load("@rules_cc//cc:cc_library.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow/compiler/mlir/tfrt:__subpackages__",
        "//tensorflow/core/tfrt:__subpackages__",
    ],
)

cc_library(
    name = "parallelization",
    srcs = ["parallelization.cc"],
    hdrs = ["parallelization.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_analysis",
        "//tensorflow/compiler/mlir/tfrt:constants",
        "//tensorflow/compiler/mlir/tfrt:cost_analysis",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "//tensorflow/core/tfrt/fallback:cost_recorder",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@tf_runtime//:stream_analysis",
    ],
)

cc_library(
    name = "assign_op_key",
    srcs = ["assign_op_key.cc"],
    hdrs = ["assign_op_key.h"],
    deps = [
        ":util",
        "//tensorflow/compiler/mlir/tfrt:constants",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "mlrt_device_constants",
    hdrs = ["mlrt_device_constants.h"],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_a_m_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops_n_z_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow/ir/host_runtime:tensorflow_tfrt_ops_inc_gen",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_to_mlrt",
    srcs = ["tf_to_mlrt.cc"],
    hdrs = ["tf_to_mlrt.h"],
    deps = [
        ":execute_op_registry",
        ":mlrt_device_constants",
        ":tpu_conversion_patterns",
        ":util",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:translate_utils",
        "//tensorflow/compiler/mlir/tensorflow/ir/host_runtime:tensorflow_tfrt_ops_inc_gen",
        "//tensorflow/compiler/mlir/tfrt:constants",
        "//tensorflow/compiler/mlir/tfrt:tfrt_pipeline_options",
        "//tensorflow/compiler/mlir/tfrt:transform_utils",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_tpu_ops",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/fallback:op_kernel_runner_cache",
        "@com_google_absl//absl/log",
        "@com_google_protobuf//:protobuf_headers",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
        "@local_xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "passes",
    srcs = ["passes.cc"],
    hdrs = ["passes.h"],
    deps = [
        ":assign_op_key",
        ":async_while",
        ":fuse_mlrt_ops",
        ":ifrt_set_tpu_host_allocator",
        ":parallelization",
        ":rewrite_ifrt_load_variable",
        ":tf_to_mlrt",
        ":while_to_map_fn",
        "//tensorflow/compiler/mlir/tfrt:tfrt_pipeline_options",
        "//tensorflow/core/tfrt/fallback:cost_recorder",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "@com_google_absl//absl/log:check",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "execute_op_registry",
    hdrs = ["execute_op_registry.h"],
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "tpu_conversion_patterns",
    srcs = ["tpu_conversion_patterns.cc"],
    hdrs = ["tpu_conversion_patterns.h"],
    deps = [
        ":execute_op_registry",
        ":mlrt_device_constants",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt:tfrt_pipeline_options",
        "//tensorflow/compiler/mlir/tfrt:transform_utils",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_tpu_ops",
        "@com_google_absl//absl/log:check",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "fuse_mlrt_ops",
    srcs = ["fuse_mlrt_ops.cc"],
    hdrs = ["fuse_mlrt_ops.h"],
    deps = [
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
    ],
)

cc_library(
    name = "import_model",
    srcs = ["import_model.cc"],
    hdrs = ["import_model.h"],
    deps = [
        ":assign_op_key",
        ":passes",
        ":while_to_map_fn",
        "//tensorflow/compiler/mlir/tensorflow:bridge_logger",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tf2xla/api/v2:tf_executor_to_graph",
        "//tensorflow/compiler/mlir/tfrt:export",
        "//tensorflow/compiler/mlir/tfrt:import_model",
        "//tensorflow/compiler/mlir/tfrt:tf_to_tfrt",
        "//tensorflow/compiler/mlir/tfrt:tfrt_compile_options",
        "//tensorflow/compiler/mlir/tfrt:tfrt_pipeline_options",
        "//tensorflow/compiler/mlir/tfrt/translate/mlrt:mlir_to_bytecode",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/tfrt/fallback:cost_recorder",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/mlrt/attribute",
        "//tensorflow/core/tfrt/mlrt/bytecode",
        "//tensorflow/core/tfrt/runtime",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "while_to_map_fn",
    srcs = ["while_to_map_fn.cc"],
    hdrs = ["while_to_map_fn.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "async_while",
    srcs = ["async_while.cc"],
    hdrs = ["async_while.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_analysis",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "rewrite_ifrt_load_variable",
    srcs = ["rewrite_ifrt_load_variable.cc"],
    hdrs = ["rewrite_ifrt_load_variable.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow/ir/host_runtime:tensorflow_tfrt_ops",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/strings",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)

cc_library(
    name = "ifrt_set_tpu_host_allocator",
    srcs = ["ifrt_set_tpu_host_allocator.cc"],
    hdrs = ["ifrt_set_tpu_host_allocator.h"],
    deps = [
        ":mlrt_device_constants",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow/ir/host_runtime:tensorflow_tfrt_ops",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:mlrt_ops",
        "//tensorflow/compiler/mlir/tfrt/ir/mlrt:tf_mlrt_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:TransformUtils",
    ],
)
