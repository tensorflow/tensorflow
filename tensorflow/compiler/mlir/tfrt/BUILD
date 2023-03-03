load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load(
    "//tensorflow/core/platform:build_config.bzl",
    "tf_platform_deps",
    "tf_proto_library",
)
load("//tensorflow:tensorflow.bzl", "if_google", "tf_cc_binary")
load("@tf_runtime//:build_defs.bzl", "tfrt_cc_library", "tfrt_cc_test")

# Note: keep the following lines separate due to the way copybara works
load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_cloud", "get_compatible_with_portable")

# TF to TFRT kernels conversion.
package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
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
        "//tensorflow/core/tfrt/graph_executor/...",
        "//tensorflow/core/tfrt/tfrt_session/...",
    ] + if_google([
        "//learning/brain/experimental/mlir/tflite/tfmrt/...",
        "//learning/brain/experimental/mlir/tfrt_compiler/...",
        "//learning/brain/experimental/tfrt/...",
        "//learning/brain/tfrt/...",
        "//learning/infra/mira/...",
        "//learning/serving/contrib/tfrt/mlir/...",
        # Allow visibility from the mlir language server.
        "//learning/brain/mlir/mlir_lsp_server/...",
        "//smartass/brain/ops/...",
        "//third_party/mlir_edge/tpgen/...",
        "//tensorflow_serving/servables/tensorflow/google/...",
        "//third_party/tf_runtime_google/...",
    ]),
)

exports_files(["run_lit.sh"])

td_library(
    name = "tf_jitrt_ops_td_files",
    srcs = ["jit/opdefs/tf_jitrt_ops.td"],
    deps = [
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_td_files",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
        "@tf_runtime//:OpBaseTdFiles",
        "@tf_runtime//:compiler_td_files",
        "@tf_runtime//backends/jitrt:jitrt_ops_td_files",
    ],
)

gentbl_cc_library(
    name = "tf_jitrt_ops_inc_gen",
    tbl_outs = [
        (
            ["-gen-op-decls"],
            "tf_jitrt_ops.h.inc",
        ),
        (
            ["-gen-op-defs"],
            "tf_jitrt_ops.cc.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "jit/opdefs/tf_jitrt_ops.td",
    deps = [":tf_jitrt_ops_td_files"],
)

tfrt_cc_library(
    name = "tf_jitrt_opdefs",
    srcs = ["jit/opdefs/tf_jitrt_ops.cc"],
    hdrs = ["jit/opdefs/tf_jitrt_ops.h"],
    deps = [
        ":tf_jitrt_ops_inc_gen",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_opdefs",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:SideEffectInterfaces",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:compiler_tfrt_op_interfaces",
        "@tf_runtime//:compiler_tfrt_traits",
        "@tf_runtime//:tensor_opdefs",
        "@tf_runtime//backends/jitrt:jitrt_opdefs",
    ],
)

tfrt_cc_library(
    name = "tf_jitrt_registration",
    srcs = ["jit/tf_jitrt_registration.cc"],
    hdrs = ["jit/tf_jitrt_registration.h"],
    deps = [
        ":tf_jitrt_opdefs",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "tf_jitrt_pipeline",
    srcs = ["jit/tf_jitrt_pipeline.cc"],
    hdrs = ["jit/tf_jitrt_pipeline.h"],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tf2xla:xla_legalize_tf",
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_jitrt_passes",
        "//tensorflow/compiler/xla/mlir/backends/cpu/transforms:passes",
        "//tensorflow/compiler/xla/mlir/runtime/ir:rt",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:compiler",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st_passes",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st_transforms",
        "//tensorflow/compiler/xla/mlir_hlo:mhlo_passes",
        "//tensorflow/compiler/xla/mlir_hlo:transforms_passes",
        "//tensorflow/compiler/xla/runtime:compiler",
        "//tensorflow/core:ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:BufferizationToMemRef",
        "@llvm-project//mlir:BufferizationTransforms",
        "@llvm-project//mlir:ComplexToStandard",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:LinalgTransforms",
        "@llvm-project//mlir:MemRefTransforms",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:ShapeToStandard",
        "@llvm-project//mlir:ShapeTransforms",
        "@llvm-project//mlir:TensorTransforms",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:VectorToSCF",
    ],
    alwayslink = 1,
)

tfrt_cc_library(
    name = "tf_jitrt",
    srcs = ["jit/tf_jitrt.cc"],
    hdrs = ["jit/tf_jitrt.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/lib/monitoring:counter",
        "//tensorflow/core/lib/monitoring:sampler",
        "//tensorflow/core/runtime_fallback/util:type_util",
        "//tensorflow/core/tfrt/utils:fallback_tensor",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:optional",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:mlir_c_runner_utils",
        "@tf_runtime//:dtype",
    ],
)

# copybara:uncomment_begin
# tfrt_cc_test(
#     name = "tf_jitrt_test",
#     srcs = ["jit/tf_jitrt_test.cc"],
#     deps = [
#         ":tf_jitrt",
#         "//tensorflow/compiler/xla/runtime:results",
#         "//tensorflow/compiler/xla/runtime:types",
#         "@com_google_googletest//:gtest_main",
#         "@llvm-project//mlir:mlir_c_runner_utils",
#         "@tf_runtime//backends/jitrt:results",
#     ],
# )
# copybara:uncomment_end

tfrt_cc_library(
    name = "tf_jitrt_query_of_death",
    hdrs = ["jit/tf_jitrt_query_of_death.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/core/platform",
    ] + tf_platform_deps(
        "tf_jitrt_query_of_death",
        platform_dir = "//tensorflow/compiler/mlir/tfrt/jit/",
    ),
)

tfrt_cc_library(
    name = "tf_jitrt_kernels",
    srcs = ["jit/tf_jitrt_kernels.cc"],
    hdrs = ["jit/tf_jitrt_kernels_registration.h"],
    alwayslink_static_registration_src = "jit/tf_jitrt_kernels_registration.cc",
    deps = [
        ":tf_jitrt",
        ":tf_jitrt_pipeline",
        ":tf_jitrt_query_of_death",
        ":tf_jitrt_request_context",
        "//tensorflow/compiler/jit:flags",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_jitrt_passes",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:compiler",
        "//tensorflow/compiler/xla/mlir/runtime/utils:async_runtime_api",
        "//tensorflow/compiler/xla/runtime:arguments",
        "//tensorflow/compiler/xla/runtime:async_runtime",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/runtime:jit_executable",
        "//tensorflow/compiler/xla/runtime:types",
        "//tensorflow/core:framework",
        "//tensorflow/core:platform_base",
        "//tensorflow/core/platform:dynamic_annotations",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//tensorflow/core/tfrt/utils:fallback_tensor",
        "@llvm-project//mlir:AsyncDialect",
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
        "@tf_runtime//backends/jitrt:jitrt_compiler",
        "@tf_runtime//backends/jitrt:results",
    ],
)

tfrt_cc_library(
    name = "tf_jitrt_request_context",
    srcs = ["jit/tf_jitrt_request_context.cc"],
    hdrs = ["jit/tf_jitrt_request_context.h"],
    deps = [
        "//tensorflow/compiler/xla/runtime:async_values_cache",
        "//tensorflow/compiler/xla/runtime:jit_executable",
        "//tensorflow/core/platform:status",
        "@tf_runtime//:hostcontext",
    ],
)

td_library(
    name = "runtime_fallback_ops_td_files",
    srcs = [
        "runtime_fallback/runtime_fallback_ops.td",
    ],
    deps = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:SideEffectInterfacesTdFiles",
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
        "@llvm-project//mlir:SideEffectInterfaces",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:tensor_opdefs",
    ],
)

cc_library(
    name = "runtime_fallback_executor",
    testonly = True,
    srcs = [
        "runtime_fallback/runtime_fallback_executor.cc",
    ],
    hdrs = [
        "runtime_fallback/runtime_fallback_executor.h",
    ],
    deps = [
        ":host_context_util",
        ":tf_to_tfrt",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:threadpool_interface",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_execute_compat",
        "//tensorflow/core/runtime_fallback/runtime:kernel_utils",
        "//tensorflow/core/runtime_fallback/runtime:runtime_fallback_alwayslink",
        "//tensorflow/core/tfrt/utils:fallback_tensor",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@tf_runtime//:bef",
        "@tf_runtime//:befexecutor",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:mlirtobef",
        "@tf_runtime//:support",
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
        ":attr_lowering_utils",
        ":transform_utils",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_analysis",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "//tensorflow/core:framework",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
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
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_async_opdefs",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_opdefs",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Transforms",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
    ],
)

cc_library(
    name = "transforms/gpu_passes",
    srcs = ["transforms/gpu_passes.cc"],
    hdrs = ["transforms/gpu_passes.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_gpu_opdefs",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
    ],
)

cc_library(
    name = "tfrt_pipeline_options",
    hdrs = ["transforms/tfrt_pipeline_options.h"],
    deps = [
        ":tfrt_compile_options",
        "@llvm-project//mlir:Pass",
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
        "transforms/remove_device_attribute.cc",
        "transforms/remove_tf_if_const_args.cc",
        "transforms/reorder_assert.cc",
        "transforms/sink_in_invariant_ops.cc",
        "transforms/tf_to_tfrt.cc",
        "transforms/tpu_passes.h",
        "transforms/xla_rewrite_pass.cc",
    ],
    hdrs = [
        "transforms/passes.h",
    ],
    deps = [
        ":attr_lowering_utils",
        ":corert_converter",
        ":cost_analysis",
        ":fallback_converter",
        ":tensor_array_side_effect_analysis",
        ":tf_jitrt_opdefs",
        ":tf_jitrt_pipeline",
        ":tfrt_pipeline_options",
        ":transforms/gpu_passes",
        ":transforms/set_shape_invariant_in_while_ops",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:FuncDialect",
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
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_jitrt_passes",
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_jitrt_clustering",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_opdefs",
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:tstring",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_async_opdefs",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_gpu_opdefs",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
        "@tf_runtime//backends/jitrt:jitrt_opdefs",
        "@tf_runtime//:stream_analysis",
        "@tf_runtime//:test_kernels_opdefs",
        ":transform_utils",
        "//tensorflow/tsl/platform:status",
    ] + if_google([
        "//learning/brain/tfrt/tpu/compiler/mlir:tf_to_tfrt_tpu",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "host_context_util",
    srcs = ["utils/host_context.cc"],
    hdrs = ["utils/host_context.h"],
    deps = [
        "//tensorflow/core/platform:logging",
        "@com_google_absl//absl/base:core_headers",
        "@tf_runtime//:hostcontext",
    ],
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
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
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
        "@llvm-project//mlir:FuncDialect",
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
        "//tensorflow/core/tfrt/graph_executor:__pkg__",
        "//tensorflow/core/tfrt/saved_model:__pkg__",
    ],
    deps = [
        ":function",
        ":tf_to_tfrt",
        ":tfrt_compile_options",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:import_model",
        "@llvm-project//mlir:FuncDialect",
        "//tensorflow/compiler/mlir/tensorflow:dump_mlir_util",
        "//tensorflow/compiler/mlir/tensorflow:error_util",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:function_body",
        "//tensorflow/core/common_runtime:function_def_utils",
        "//tensorflow/core/tfrt/fallback:fallback_state",
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
    deps = ["@com_google_absl//absl/strings"],
)

cc_library(
    name = "cost_analysis",
    srcs = ["analysis/cost_analysis.cc"],
    hdrs = ["analysis/cost_analysis.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tfrt/fallback:cost_recorder",
        "//tensorflow/core/tfrt/fallback:op_cost_map_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "test_cost_analysis_pass",
    srcs = ["analysis/test_cost_analysis_pass.cc"],
    deps = [
        ":cost_analysis",
        "@llvm-project//mlir:FuncDialect",
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
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "test_tensor_array_side_effect_analysis",
    srcs = ["analysis/test_tensor_array_side_effect_analysis.cc"],
    deps = [
        ":tensor_array_side_effect_analysis",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
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
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_jitrt_passes",
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_jitrt_test_passes",
        "//tensorflow/compiler/mlir/tfrt:tf_to_tfrt",
    ] + if_google([
        "//learning/brain/tfrt/tpu/compiler/mlir:tf_to_tfrt_tpu",
    ]),
)

cc_library(
    name = "transforms/set_shape_invariant_in_while_ops",
    srcs = ["transforms/set_shape_invariant_in_while_ops.cc"],
    hdrs = ["transforms/set_shape_invariant_in_while_ops.h"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
)

tf_cc_binary(
    name = "tf-tfrt-opt",
    testonly = True,
    srcs = ["tf-tfrt-opt.cc"],
    deps = [
        ":passes",
        ":test_cost_analysis_pass",
        ":test_tensor_array_side_effect_analysis",
        ":tf_jitrt_opdefs",
        ":tf_to_tfrt",
        "//tensorflow/compiler/mlir:init_mlir",
        "//tensorflow/compiler/mlir:passes",
        "//tensorflow/compiler/mlir/lite:tensorflow_lite",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:bridge_pass_test_pipeline_registration",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_passes",
        "//tensorflow/compiler/mlir/tfrt:transforms/gpu_passes",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_async_opdefs",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_opdefs",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_sync_opdefs",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_gpu_opdefs",
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_jitrt_passes",
        "//tensorflow/compiler/mlir/tfrt/jit/transforms:tf_jitrt_test_passes",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st",
        "//tensorflow/compiler/xla/mlir_hlo:gml_st_passes",
        "//tensorflow/core:lib",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:MlirOptLib",
        "@llvm-project//mlir:ShapeDialect",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//:print_stream_pass",
        "@tf_runtime//backends/jitrt:jitrt_compiler",
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
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tf_graph_optimization_pass",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo",
        "//tensorflow/compiler/xla/mlir_hlo:lhlo_gpu",
        "@llvm-project//mlir:AllPassesAndDialects",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:MlirOptLib",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//backends/gpu:gpu_opdefs",
        "@tf_runtime//backends/gpu:gpu_passes",
    ],
)

tf_cc_binary(
    name = "tfrt_translate",
    srcs = ["tools/tfrt_translate/static_registration.cc"],
    visibility = [":friends"],
    deps = [
        ":tf_jitrt_registration",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:TranslateLib",
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
        ":tf_jitrt_kernels_alwayslink",
        "@tf_runtime//:dtype",
        "@tf_runtime//:simple_tracing_sink",
        "@tf_runtime//tools:bef_executor_expensive_kernels",
        "@tf_runtime//tools:bef_executor_jit_kernels",
        "@tf_runtime//tools:bef_executor_lib",
        "@tf_runtime//tools:bef_executor_lightweight_kernels",
    ],
)

cc_library(
    name = "tfrt_fallback_registration",
    srcs = [
        "tfrt_fallback_registration.cc",
    ],
    hdrs = [
        "tfrt_fallback_registration.h",
    ],
    visibility = [":friends"] + if_google([
        "//learning/brain/experimental/tfrt/visualization:__pkg__",
        # Allow visibility from the mlir language server.
        "//learning/brain/mlir/mlir_lsp_server:__pkg__",
    ]),
    deps = [
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_async_opdefs",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_opdefs",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_sync_opdefs",
        "@llvm-project//mlir:IR",
    ],
)

tf_cc_binary(
    name = "tfrt_fallback_translate",
    srcs = [
        "tfrt_fallback_translate_registration.cc",
    ],
    visibility = [":friends"],
    deps = [
        "@llvm-project//mlir:TranslateLib",
        "//tensorflow/compiler/mlir/tfrt/ir:tfrt_fallback_opdefs",
        "//tensorflow/compiler/mlir/tfrt:tf_jitrt_registration",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tfrt:tfrt_fallback_registration",
        "@tf_runtime//:init_tfrt_dialects",
        "@tf_runtime//:mlirtobef_translate",
    ] + if_google(
        ["//third_party/tf_runtime_llvm:tfrt_translate_main"],
        ["@tf_runtime//third_party/llvm_derived:tfrt_translate_main"],
    ),
)

cc_library(
    name = "attr_lowering_utils",
    srcs = [
        "transforms/attr_lowering_utils.cc",
    ],
    hdrs = [
        "transforms/attr_lowering_utils.h",
    ],
    visibility = [":friends"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_attributes",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_types",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:IR",
        "@tf_runtime//:core_runtime_opdefs",
    ],
)

cc_library(
    name = "transform_utils",
    srcs = [
        "transforms/utils.cc",
    ],
    hdrs = [
        "transforms/utils.h",
    ],
    visibility = [":friends"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/compiler/mlir/tensorflow:tensorflow_ops",
        "@com_google_absl//absl/strings",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@tf_runtime//:basic_kernels_opdefs",
        "@tf_runtime//:core_runtime_opdefs",
    ],
)

cc_library(
    name = "transforms/update_op_cost_in_tfrt_mlir",
    srcs = ["transforms/update_op_cost_in_tfrt_mlir.cc"],
    hdrs = ["transforms/update_op_cost_in_tfrt_mlir.h"],
    deps = [
        ":cost_analysis",
        "//tensorflow/core/tfrt/fallback:cost_recorder",
        "@llvm-project//mlir:IR",
    ],
)
