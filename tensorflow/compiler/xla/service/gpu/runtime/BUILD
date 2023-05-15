load("//tensorflow/tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")
load("//tensorflow/compiler/xla:xla.bzl", "xla_cc_test")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load(
    "//tensorflow/tsl/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/xla:friends",
    ],
)

cc_library(
    name = "cholesky",
    srcs = ["cholesky.cc"],
    hdrs = ["cholesky.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:cholesky_thunk",
        "//tensorflow/compiler/xla/service/gpu:gpu_asm_opts_util",
    ],
)

cc_library(
    name = "collectives",
    srcs = ["collectives.cc"],
    hdrs = ["collectives.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:computation_placer_hdr",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:global_device_id",
        "//tensorflow/compiler/xla/service/gpu:gpu_executable_run_options",
        "//tensorflow/compiler/xla/service/gpu:nccl_collective_thunks",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/compiler/xla/stream_executor:executor_cache",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
)

cc_library(
    name = "conv",
    srcs = ["conv.cc"],
    hdrs = ["conv.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:custom_call_encoding",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:gpu_asm_opts_util",
        "//tensorflow/compiler/xla/service/gpu:gpu_conv_algorithm_picker",
        "//tensorflow/compiler/xla/service/gpu:gpu_conv_runner",
        "//tensorflow/compiler/xla/service/gpu:gpu_serializable_autotuner",
        "//tensorflow/compiler/xla/service/gpu:non_atomically_upgradeable_rw_lock",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "//tensorflow/compiler/xla/stream_executor:device_memory_allocator",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:attribute_exporter",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "conv_reorder",
    srcs = ["conv_reorder.cc"],
    hdrs = ["conv_reorder.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:executable",
    ],
)

cc_library(
    name = "custom_call",
    srcs = ["custom_call.cc"],
    hdrs = ["custom_call.h"],
    deps = [
        ":support",
        ":triangular_solve",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:custom_call_status_internal",
        "//tensorflow/compiler/xla/service:custom_call_status_public_headers",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:cublas_cudnn",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream_header",
    ],
)

cc_library(
    name = "executable",
    srcs = ["executable.cc"],
    hdrs = ["executable.h"],
    copts = if_cuda_is_configured(["-DGOOGLE_CUDA=1"]),
    deps = [
        ":cholesky",
        ":collectives",
        ":concurrent_region",
        ":conv",
        ":conv_reorder",
        ":cublas_lt_matmul",
        ":custom_call",
        ":fft",
        ":gemm",
        ":graph_launch",
        ":io_feed",
        ":kernel_launch",
        ":memcpy",
        ":memset",
        ":send_recv",
        ":support",
        ":topk",
        ":tracing",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:compilation_pipeline_gpu",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/runtime:ffi",
        "//tensorflow/compiler/xla/runtime:jit_executable",
        "//tensorflow/compiler/xla/runtime:module_registry",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:buffer_allocations",
        "//tensorflow/compiler/xla/service/gpu:non_atomically_upgradeable_rw_lock",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream",
        "//tensorflow/tsl/protobuf:dnn_proto_cc",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "fft",
    srcs = ["fft.cc"],
    hdrs = ["fft.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:custom_call_encoding",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/runtime:state",
        "//tensorflow/compiler/xla/service/gpu:fft_thunk",
        "//tensorflow/compiler/xla/stream_executor:fft",
        "//tensorflow/compiler/xla/translate/mhlo_to_hlo:attribute_exporter",
    ],
)

cc_library(
    name = "topk_kernel",
    srcs = if_cuda_is_configured(
        [
            "topk_kernel.cc",
        ],
    ),
    hdrs = if_cuda_is_configured(["topk_kernel.h"]),
    compatible_with = [],
    deps = [
        ":topk_kernel_cuda",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/stream_executor:platform",
        "//tensorflow/compiler/xla/stream_executor:stream_executor_headers",  # build_cleaner: keep
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_types_header",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

cuda_library(
    name = "topk_kernel_cuda",
    srcs = if_cuda_is_configured(
        [
            "topk_kernel.cu.cc",
        ],
    ),
    hdrs = if_cuda_is_configured(["topk_kernel_common.h"]),
    compatible_with = [],
    deps = [
        "//third_party/eigen3",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

xla_cc_test(
    name = "topk_kernel_test",
    srcs = if_cuda_is_configured(["topk_kernel_test.cc"]),
    linkstatic = 1,
    tags = tf_cuda_tests_tags(),
    deps = [
        ":topk_kernel",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_types_header",
        "//tensorflow/compiler/xla/stream_executor/host:host_platform",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_benchmark",
        "//tensorflow/tsl/platform:test_main",
        "//third_party/eigen3",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/strings",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)

xla_cc_test(
    name = "topk_test",
    srcs = if_cuda_is_configured(["topk_test.cc"]),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":topk",
        "//tensorflow/compiler/xla:error_spec",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/service:gpu_plugin",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:platform_util",
        "//tensorflow/compiler/xla/service:topk_rewriter",
        "//tensorflow/compiler/xla/service/gpu:topk_specializer",
        "//tensorflow/compiler/xla/tests:hlo_test_base",
        "//tensorflow/compiler/xla/tests:verified_hlo_module",
        "//tensorflow/compiler/xla/tests:xla_internal_test_main",  # fixdeps: keep
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "topk",
    srcs = if_cuda_is_configured(
        ["topk.cc"],
        ["topk_no_cuda.cc"],
    ),
    hdrs = ["topk.h"],
    deps = if_cuda_is_configured([":topk_kernel"]) + [
        ":support",
        "//tensorflow/compiler/xla:executable_run_options",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:xla_data_proto_cc",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/hlo/ir:hlo",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:custom_call_encoding",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/runtime:state",
        "//tensorflow/compiler/xla/runtime/ffi:ffi_api",
        "//tensorflow/compiler/xla/runtime/ffi:ffi_c_api_hdrs",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo_pass",
        "//tensorflow/compiler/xla/service:tuple_util",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_stream_header",
        "//tensorflow/compiler/xla/stream_executor/gpu:gpu_types_header",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "gemm",
    srcs = ["gemm.cc"],
    hdrs = ["gemm.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/runtime:state",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:hlo_module_config",
        "//tensorflow/compiler/xla/service/gpu:matmul_utils",
        "//tensorflow/compiler/xla/service/gpu:non_atomically_upgradeable_rw_lock",
        "//tensorflow/compiler/xla/stream_executor:blas",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "@com_google_absl//absl/container:node_hash_map",
    ] + if_cuda_is_configured([
        "//tensorflow/compiler/xla/service/gpu:gemm_algorithm_picker",
        "//tensorflow/compiler/xla/stream_executor/gpu:redzone_allocator",
    ]),
)

cc_library(
    name = "graph_launch",
    srcs = ["graph_launch.cc"],
    hdrs = ["graph_launch.h"],
    deps = [
        ":concurrent_region",
        ":conv",
        ":gemm",
        ":kernel_launch",
        ":support",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/runtime:state",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:non_atomically_upgradeable_rw_lock",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/tsl/profiler/lib:scoped_annotation_stack",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/synchronization",
    ] + if_cuda_is_configured([
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_graph",
    ]),
)

cc_library(
    name = "concurrent_region",
    srcs = ["concurrent_region.cc"],
    hdrs = ["concurrent_region.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service:stream_pool",
        "//tensorflow/compiler/xla/stream_executor",
        "//tensorflow/tsl/platform:statusor",
    ],
)

cc_library(
    name = "io_feed",
    srcs = ["io_feed.cc"],
    hdrs = ["io_feed.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:io_feed_manager",
    ],
)

cc_library(
    name = "kernel_launch",
    srcs = ["kernel_launch.cc"],
    hdrs = ["kernel_launch.h"],
    deps = [
        ":concurrent_region",
        ":support",
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/runtime:state",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:launch_dimensions",
        "//tensorflow/compiler/xla/service/gpu:stream_executor_util",
        "//tensorflow/compiler/xla/stream_executor",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/synchronization",
    ] + if_cuda_is_configured([
        "//tensorflow/compiler/xla/stream_executor/cuda:cuda_graph",
    ]),
)

cc_library(
    name = "cublas_lt_matmul",
    srcs = ["cublas_lt_matmul.cc"],
    hdrs = ["cublas_lt_matmul.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:custom_call_encoding",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/runtime:logical_result",
        "//tensorflow/compiler/xla/runtime:state",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:matmul_utils",
        "//tensorflow/compiler/xla/stream_executor:scratch_allocator",
        "//tensorflow/compiler/xla/stream_executor/cuda:cublas_lt_header",
        "//tensorflow/tsl/platform:status",
    ],
)

cc_library(
    name = "memcpy",
    srcs = ["memcpy.cc"],
    hdrs = ["memcpy.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:io_feed_manager",
    ],
)

cc_library(
    name = "memset",
    srcs = ["memset.cc"],
    hdrs = ["memset.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:io_feed_manager",
    ],
)

cc_library(
    name = "support",
    hdrs = ["support.h"],
    deps = [
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:custom_call_encoding",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/service/gpu:matmul_utils",
        "//tensorflow/compiler/xla/stream_executor:blas",
        "//tensorflow/compiler/xla/stream_executor:device_memory",
        "@llvm-project//llvm:Support",
    ],
)

cc_library(
    name = "send_recv",
    srcs = ["send_recv.cc"],
    hdrs = ["send_recv.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla/mlir/runtime/transforms:custom_call_encoding",
        "//tensorflow/compiler/xla/mlir_hlo",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/stream_executor:event",
        "//tensorflow/tsl/profiler/lib:traceme",
        "//tensorflow/tsl/profiler/lib:traceme_encode",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@tf_runtime//:async_value",
    ],
)

cc_library(
    name = "tracing",
    srcs = ["tracing.cc"],
    hdrs = ["tracing.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:custom_call_registry",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/runtime:tracing",
        "//tensorflow/compiler/xla/runtime:type_id",
        "//tensorflow/tsl/profiler/lib:scoped_annotation_stack",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "triangular_solve",
    srcs = ["triangular_solve.cc"],
    hdrs = ["triangular_solve.h"],
    deps = [
        ":support",
        "//tensorflow/compiler/xla:xla_proto_cc",
        "//tensorflow/compiler/xla/runtime:custom_call",
        "//tensorflow/compiler/xla/runtime:executable",
        "//tensorflow/compiler/xla/service:executable",
        "//tensorflow/compiler/xla/service/gpu:gpu_asm_opts_util",
        "//tensorflow/compiler/xla/service/gpu:triangular_solve_thunk",
        "//tensorflow/tsl/platform:human_readable_json",
    ],
)
