load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("//tensorflow/core/platform:build_config.bzl", "tf_proto_library")
load("//tensorflow:tensorflow.bzl", "if_google", "tf_cc_binary")
load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library")

# TF to TFRT kernels conversion.
package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/...",
        "//tensorflow/core/runtime_fallback/...",
        "//tensorflow/core/tfrt/eager/...",
        "//tensorflow/core/tfrt/experimental/data/...",
        "//tensorflow/core/tfrt/saved_model/...",
    ] + if_google([
        "//learning/brain/experimental/mlir/tflite/tfmrt/...",
        "//learning/brain/experimental/mlir/tfrt_compiler/...",
        "//learning/brain/experimental/tfrt/...",
        "//learning/brain/tfrt/...",
        "//learning/serving/contrib/tfrt/mlir/...",
        # Allow visibility from the mlir language server.
        "//learning/brain/mlir/mlir_lsp_server/...",
        "//platforms/xla/tests/gpu/...",
        "//smartass/brain/ops/...",
        "//tensorflow_serving/servables/tensorflow/google/...",
        "//third_party/tf_runtime_google/...",
    ]),
)

exports_files(["run_lit.sh"])

td_library(
    name = "tf_cpurt_ops_td_files",
    srcs = ["jit/opdefs/tf_cpurt_ops.td"],
    deps = [
        "//tensorflow/core/runtime_fallback/opdefs:tfrt_fallback_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@tf_runtime//:OpBaseTdFiles",
        "@tf_runtime//:compiler_td_files",
        "@tf_runtime//backends/cpu:cpurt_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tf_cpurt_ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "tf_cpurt_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "tf_cpurt_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "jit/opdefs/tf_cpurt_ops.td",
    deps = [":tf_cpurt_ops_td_files"],
)

tfrt_cc_library(
    name = "tf_cpurt_opdefs",
    srcs = ["jit/opdefs/tf_cpurt_ops.cc"],
    hdrs = ["jit/opdefs/tf_cpurt_ops.h"],
    deps = [
        ":tf_cpurt_ops_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/core/runtime_fallback/opdefs:tfrt_fallback_opdefs",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@llvm-project//mlir:StandardOps",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:compiler_tfrt_op_interfaces",
        "@tf_runtime//:compiler_tfrt_traits",
        "@tf_runtime//:tensor_opdefs",
        "@tf_runtime//backends/cpu:cpurt_opdefs",
    ],
)

tfrt_cc_library(
    name = "tf_cpurt_registration",
    srcs = ["jit/tf_cpurt_registration.cc"],
    hdrs = ["jit/tf_cpurt_registration.h"],
    deps = [
        ":tf_cpurt_opdefs",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_cpurt_pipeline",
    srcs = ["jit/tf_cpurt_pipeline.cc"],
    hdrs = ["jit/tf_cpurt_pipeline.h"],
    deps = [
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/mlir/hlo",
        "//tensorflow/compiler/mlir/hlo:all_passes",
        "//tensorflow/compiler/mlir/hlo:legalize_to_linalg",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_cpurt_passes",
        "//tensorflow/compiler/mlir/tools/kernel_gen/transforms:passes",
        "//tensorflow/compiler/mlir/xla:xla_legalize_tf",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Async",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeToStandard",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:StandardOpsTransforms",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorToSCF",
    ],
    alwayslink = 1,
)

tfrt_cc_library(
    name = "tf_cpurt",
    hdrs = ["jit/tf_cpurt.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/runtime_fallback/util:type_util",
        "//tensorflow/core/tfrt/utils:fallback_tensor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:mlir_c_runner_utils",
        "@tf_runtime//:dtype",
        "@tf_runtime//backends/cpu:cpurt",
    ],
)

tfrt_cc_library(
    name = "tf_cpurt_kernels",
    srcs = ["jit/tf_cpurt_kernels.cc"],
    alwayslink_static_registration_src = "jit/tf_cpurt_kernels_registration.cc",
    deps = [
        ":tf_cpurt",
        ":tf_cpurt_pipeline",
        ":tf_cpurt_request_context",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_cpurt_passes",
        "//tensorflow/core:framework",
        "//tensorflow/core:platform_base",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//tensorflow/core/tfrt/utils:fallback_tensor",
        "@llvm-project//mlir:Async",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:mlir_async_runtime_api",
        "@tf_runtime//:dtype",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
        "@tf_runtime//:tensor",
        "@tf_runtime//:tracing",
        "@tf_runtime//backends/cpu:async_runtime",
        "@tf_runtime//backends/cpu:async_runtime_api",
        "@tf_runtime//backends/cpu:cpurt",
    ],
)

tfrt_cc_library(
    name = "tf_cpurt_request_context",
    srcs = ["jit/tf_cpurt_request_context.cc"],
    hdrs = ["jit/tf_cpurt_request_context.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:gce"],
    deps = [
        "//tensorflow/core/platform:status",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//backends/cpu:cpurt",
    ],
)

td_library(
    name = "runtime_fallback_ops_td_files",
    srcs = [
        "runtime_fallback/runtime_fallback_ops.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectTdFiles",
        "@tf_runtime//:OpBaseTdFiles",
    ],
)

gentbl_cc_library(
    name = "runtime_fallback_ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "runtime_fallback_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "runtime_fallback_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "runtime_fallback/runtime_fallback_ops.td",
    deps = [":runtime_fallback_ops_td_files"],
)

cc_library(
    name = "runtime_fallback_opdefs",
    srcs = [
        "runtime_fallback/runtime_fallback_combine.cc",
        "runtime_fallback/runtime_fallback_ops.cc",
    ],
    hdrs = [
        "runtime_fallback/runtime_fallback_ops.h",
    ],
    deps = [
        ":runtime_fallback_ops_inc_gen",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffects",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:tensor_opdefs",
    ],
)

cc_library(
    name = "corert_converter",
    srcs = [
        "transforms/corert_converter.cc",
    ],
    hdrs = [
        "transforms/corert_converter.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_analysis",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:framework",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
        "@tf_runtime//:distributed_kernels_opdefs",
    ],
)

cc_library(
    name = "fallback_converter",
    srcs = [
        "transforms/fallback_converter.cc",
    ],
    hdrs = [
        "transforms/fallback_converter.h",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core/runtime_fallback/opdefs:tfrt_fallback_async_opdefs",
        "//tensorflow/core/runtime_fallback/opdefs:tfrt_fallback_opdefs",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
    ],
)

cc_library(
    name = "tf_to_tfrt",
    srcs = [
        "transforms/cross_device_transfer.cc",
        "transforms/deduplicate_batch_function.cc",
        "transforms/fuse_tpu_compile_and_execute_ops.cc",
        "transforms/insert_tensor_copy.cc",
        "transforms/lower_saved_model.cc",
        "transforms/merge_tf_if_ops.cc",
        "transforms/optimize.cc",
        "transforms/optimize_tf_control_flow_side_effect.cc",
        "transforms/remote_run_encapsulate.cc",
        "transforms/remove_device_attribute.cc",
        "transforms/remove_tf_if_const_args.cc",
        "transforms/reorder_assert.cc",
        "transforms/set_shape_invariant_in_while_ops.cc",
        "transforms/tf_to_tfrt.cc",
        "transforms/tpu_passes.h",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":corert_converter",
        ":cost_analysis",
        ":fallback_converter",
        ":tensor_array_side_effect_analysis",
        ":tf_cpurt_opdefs",
        ":tf_cpurt_pipeline",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:bridge_logger",
        "//tensorflow/compiler/mlir/tensorflow:convert_tensor",
        "//tensorflow/compiler/mlir/tensorflow:device_util",
        "//tensorflow/compiler/mlir/tensorflow:serialize_mlir_module_utils",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_analysis",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_op_interfaces",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_cpurt_passes",
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_cpurt_clustering",
        "//tensorflow/core/runtime_fallback/opdefs:tfrt_fallback_opdefs",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:tstring",
        "//tensorflow/core/runtime_fallback:tfrt_fallback_async_opdefs",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
        "@tf_runtime//:distributed_kernels_opdefs",
        "@tf_runtime//backends/cpu:cpurt_opdefs",
        "@tf_runtime//:stream_analysis",
        "@tf_runtime//:test_kernels_opdefs",
    ] + if_google([
        "//learning/brain/tfrt/tpu/compiler/mlir:tf_to_tfrt_tpu",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "tf_to_tfrt_data",
    srcs = [
        "transforms/tf_to_tfrt_data.cc",
    ],
    hdrs = [
        "transforms/tf_to_tfrt_data.h",
    ],
    deps = [
        ":tf_to_tfrt",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:mlir_roundtrip_flags",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:bef",
        "@tf_runtime//:data_opdefs",
        "@tf_runtime//:mlirtobef",
    ],
    alwayslink = 1,
)

cc_library(
    name = "function",
    srcs = [
        "function/function.cc",
    ],
    hdrs = [
        "function/function.h",
    ],
    deps = [
        ":tf_to_tfrt",
        ":tfrt_compile_options",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "@tf_runtime//:bef",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:mlirtobef",
        "@tf_runtime//:tensor",
    ],
)

cc_library(
    name = "saved_model",
    srcs = [
        "saved_model/saved_model.cc",
    ],
    hdrs = [
        "saved_model/saved_model.h",
    ],
    deps = [
        ":tf_to_tfrt",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:StandardOps",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:convert_type",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tensorflow:tf_dialect_passes",
        "//tensorflow/compiler/mlir/tensorflow:translate_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:status",
        "@tf_runtime//:bef",
        "@tf_runtime//:core_runtime",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:mlirtobef",
        "@tf_runtime//:tensor",
    ] + if_google([
        "//learning/brain/tfrt/tpu/compiler/mlir:tf_to_tfrt_tpu",
    ]),
)

cc_library(
    name = "import_model",
    srcs = [
        "translate/import_model.cc",
    ],
    hdrs = [
        "translate/import_model.h",
    ],
    visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt/visualization:__pkg__",
        "//tensorflow/compiler/mlir/tfrt/tests/saved_model:__pkg__",
        "//tensorflow/core/tfrt/eager:__pkg__",
        "//tensorflow/core/tfrt/saved_model:__pkg__",
        "@tf_runtime//:__subpackages__",
    ],
    deps = [
        ":function",
        ":tf_to_tfrt",
        ":tfrt_compile_options",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:function_body",
        "//tensorflow/core/common_runtime:function_def_utils",
        "//tensorflow/core/platform:status",
        "@tf_runtime//:bef",
        "@tf_runtime//:mlirtobef",
    ] + if_google([
        "//learning/brain/tfrt/tpu/compiler/mlir:tf_to_tfrt_tpu",
    ]),
)

cc_library(
    name = "tfrt_compile_options",
    srcs = ["translate/tfrt_compile_options.cc"],
    hdrs = ["translate/tfrt_compile_options.h"],
)

cc_library(
    name = "convert_xla_gpu",
    srcs = [
        "translate/convert_xla_gpu.cc",
    ],
    hdrs = [
        "translate/convert_xla_gpu.h",
    ],
    tags = [
        "gpu",
        "no_oss",
    ],
    visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt/jax:__subpackages__",
        "//platforms/xla/tests/gpu:__pkg__",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu:pass_utils",
        "//tensorflow/compiler/mlir/xla:mhlo_to_lhlo_with_xla",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//:mlirtobef_translate",
        "@tf_runtime//backends/gpu:gpu_opdefs",
        "@tf_runtime//backends/gpu:gpu_system",
    ],
)

cc_library(
    name = "cost_analysis",
    srcs = ["analysis/cost_analysis.cc"],
    hdrs = ["analysis/cost_analysis.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "test_cost_analysis_pass",
    srcs = ["analysis/test_cost_analysis_pass.cc"],
    deps = [
        ":cost_analysis",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tensor_array_side_effect_analysis",
    srcs = ["analysis/tensor_array_side_effect_analysis.cc"],
    hdrs = ["analysis/tensor_array_side_effect_analysis.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "test_tensor_array_side_effect_analysis",
    srcs = ["analysis/test_tensor_array_side_effect_analysis.cc"],
    deps = [
        ":tensor_array_side_effect_analysis",
        "@llvm-project//mlir:Pass",
    ],
    alwayslink = 1,
)

cc_library(
    name = "compatibility_analysis",
    srcs = [
        "analysis/compatibility_analysis.cc",
    ],
    hdrs = [
        "analysis/compatibility_analysis.h",
    ],
    deps = [
        ":analysis/analysis_proto_cc",
        ":tf_to_tfrt",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:lib_proto_parsing",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:StandardOps",
        "@llvm-project//mlir:Translation",
    ],
    alwayslink = 1,
)

tf_proto_library(
    name = "analysis/analysis_proto",
    srcs = ["analysis/analysis.proto"],
    cc_api_version = 2,
)

cc_library(
    name = "passes",
    visibility = [
        ":__subpackages__",
    ],
    deps = [
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_cpurt_passes",
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_cpurt_test_passes",
        "//tensorflow/compiler/mlir/tfrt:tf_to_tfrt",
        "//tensorflow/compiler/mlir/tfrt:tf_to_tfrt_data",
    ] + if_google([
        "//learning/brain/tfrt/tpu/compiler/mlir:tf_to_tfrt_tpu",
    ]),
)

tf_cc_binary(
    name = "tf-tfrt-opt",
    testonly = True,
    srcs = ["tf-tfrt-opt.cc"],
    deps = [
        ":passes",
        ":test_cost_analysis_pass",
        ":test_tensor_array_side_effect_analysis",
        ":tf_to_tfrt",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir:passes",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:bridge_pass_test_pipeline_registration",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tfrt:tf_cpurt_opdefs",
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_cpurt_passes",
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_cpurt_test_passes",
        "//tensorflow/core:lib",
        "//tensorflow/core/runtime_fallback/opdefs:tfrt_fallback_async_opdefs",
        "//tensorflow/core/runtime_fallback/opdefs:tfrt_fallback_opdefs",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:Shape",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//:print_stream_pass",
    ],
)

tf_cc_binary(
    name = "lhlo-tfrt-opt",
    srcs = ["lhlo-tfrt-opt.cc"],
    tags = [
        "gpu",
        "no_oss",
    ],
    visibility = [":friends"],
    deps = [
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir:passes",
        "//tensorflow/compiler/mlir/hlo:lhlo",
        "//tensorflow/compiler/mlir/hlo:lhlo_gpu",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tf_graph_optimization_pass",
        "//tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu:lmhlo_to_gpu_binary",
        "//tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu:lmhlo_to_tfrt_gpu",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//backends/gpu:gpu_opdefs",
    ],
)

tf_cc_binary(
    name = "tfrt_translate",
    srcs = ["tools/tfrt_translate/static_registration.cc"],
    visibility = [":friends"],
    deps = [
        ":tf_cpurt_registration",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Translation",
        "@tf_runtime//:beftomlir_translate",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//:mlirtobef_translate",
    ] + if_google(
        ["//third_party/tf_runtime_llvm:tfrt_translate_main"],
        ["@tf_runtime//third_party/llvm_derived:tfrt_translate_main"],
    ),
)

tf_cc_binary(
    name = "bef_executor",
    testonly = True,
    visibility = [":friends"],
    deps = [
        ":tf_cpurt_kernels_alwayslink",
        "@tf_runtime//:dtype",
        "@tf_runtime//:simple_tracing_sink",
        "@tf_runtime//tools:bef_executor_expensive_kernels",
        "@tf_runtime//tools:bef_executor_jit_kernels",
        "@tf_runtime//tools:bef_executor_lib",
        "@tf_runtime//tools:bef_executor_lightweight_kernels",
    ],
)
