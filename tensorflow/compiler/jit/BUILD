load("@local_xla//xla/stream_executor:build_defs.bzl", "if_cuda_or_rocm")
load("//tensorflow:tensorflow.bzl", "if_libtpu", "if_with_tpu_support", "tf_cc_test", "tf_copts", "tf_cuda_cc_test", "tf_cuda_only_cc_test")
load("//tensorflow:tensorflow.default.bzl", "cc_header_only_library", "filegroup", "tf_custom_op_py_strict_library")
load("//tensorflow/core/platform:build_config.bzl", "tf_additional_all_protos", "tf_proto_library")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "if_static",
    "tf_cuda_tests_tags",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        ":internal",
        "//third_party/cloud_tpu/inference_converter:__pkg__",
    ],
    licenses = ["notice"],
)

package_group(
    name = "internal",
    includes = [
        "//tensorflow/compiler/tf2xla:internal",
    ],
    packages = [
        "//tensorflow/c/...",
        "//tensorflow/compiler/tests/...",
        "//tensorflow/core/tfrt/saved_model/...",
        "//tensorflow/python/...",
    ],
)

package_group(
    name = "friends",
    includes = [
        "//tensorflow/compiler/tf2xla:friends",
    ],
)

# defs.cc/h only contains string constants, and can be included in mobile
# builds.
filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "defs.cc",
        "defs.h",
    ],
    visibility = [":friends"],
)

# Target that bundles up the XLA CPU and GPU JIT devices.
cc_library(
    name = "jit",
    visibility = [
        ":friends",
        "//learning/tfx:__subpackages__",
    ],
    deps = [
        ":xla_cpu_device",
        ":xla_cpu_jit",
        "//tensorflow/compiler/plugin",
    ] + if_cuda_or_rocm([
        ":xla_gpu_device",
        ":xla_gpu_jit",
    ]) + if_with_tpu_support([
        ":xla_tpu_device",
        ":xla_tpu_jit",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "xla_cpu_jit",
    visibility = ["//visibility:public"],
    deps = [
        ":jit_compilation_passes",
        ":xla_kernel_creator",  # buildcleaner: keep
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_dummy_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core/tfrt/common:pjrt_cpu_client_registration",
    ] + if_with_tpu_support(
        if_false = ["@local_xla//xla/service:cpu_plugin"],
        if_true = [],
    ),
    alwayslink = 1,
)

cc_library(
    name = "xla_gpu_jit",
    visibility = ["//visibility:public"],
    deps = if_cuda_or_rocm([
        ":jit_compilation_passes",
        ":xla_kernel_creator",  # buildcleaner: keep
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_dummy_ops",
        "@local_xla//xla/service:gpu_plugin",
        "//tensorflow/core/tfrt/common:pjrt_gpu_client_registration",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "xla_tpu_jit",
    visibility = ["//visibility:public"],
    deps = if_libtpu([
        "//tensorflow/core/tpu/graph_rewrite:tpu_rewrite_pass_registration",
        "//tensorflow/core/tpu/graph_rewrite:configure_tpu_embedding_rewrite_registration",
        "@local_xla//xla/stream_executor/tpu:tpu_transfer_manager",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "xla_cpu_device",
    srcs = ["xla_cpu_device.cc"],
    visibility = [":friends"],
    deps = [
        ":common",
        ":flags",
        ":jit_compilation_passes",
        ":xla_device",
        ":xla_kernel_creator",  # buildcleaner: keep
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@local_xla//xla/stream_executor:platform_manager",
    ] + if_libtpu(
        if_false = [
            "@local_xla//xla/service:cpu_plugin",  # buildcleaner: keep
        ],
        if_true = [],
    ),
    alwayslink = 1,
)

cc_library(
    name = "xla_gpu_device",
    srcs = ["xla_gpu_device.cc"],
    visibility = [":friends"],
    deps = [
        ":common",
        ":flags",
        ":jit_compilation_passes",
        ":xla_device",
        ":xla_device_no_jit_rewrite_registration",
        ":xla_kernel_creator",  # buildcleaner: keep
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/stream_executor:platform_manager",
        "@local_xla//xla/stream_executor/gpu:gpu_init",
    ] + if_libtpu(
        if_false = [
            "@local_xla//xla/service:gpu_plugin",  # buildcleaner: keep
        ],
        if_true = [],
    ),
    alwayslink = 1,
)

cc_library(
    name = "xla_tpu_device",
    srcs = ["xla_tpu_device.cc"],
    hdrs = ["xla_tpu_device.h"],
    visibility = [":friends"],
    deps = [
        ":xla_device",
        ":xla_device_context",
        ":xla_kernel_creator",  # buildcleaner: keep
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:device_factory",
        "//tensorflow/core/common_runtime:dma_helper",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/tpu:tpu_defs",
        "//tensorflow/core/tpu:tpu_node_device_util",
        "//tensorflow/core/tpu:virtual_device",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/stream_executor/tpu:c_api_conversions",
        "@local_xla//xla/stream_executor/tpu:status_helper",
        "@local_xla//xla/stream_executor/tpu:tpu_api",
        "@local_xla//xla/stream_executor/tpu:tpu_executor_base",
        "@local_xla//xla/stream_executor/tpu:tpu_node_context",
        "@local_xla//xla/stream_executor/tpu:tpu_platform_interface",
        "@local_xla//xla/stream_executor/tpu:tpu_stream_interface",
    ] + if_static([
        "//tensorflow/core/common_runtime:copy_tensor",
        ":jit_compilation_passes",
    ]),
    alwayslink = 1,
)

cc_library(
    name = "xla_tensor",
    srcs = ["xla_tensor.cc"],
    hdrs = ["xla_tensor.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/memory",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/service:shaped_buffer",
    ],
)

XLA_DEVICE_DEPS = [
    ":common",
    ":pjrt_device_context",
    ":variable_info",
    ":variable_info_util",
    ":xla_compile_util",
    ":xla_launch_util",
    ":xla_tensor",
    "@com_google_absl//absl/base",
    "@com_google_absl//absl/memory",
    "@com_google_absl//absl/strings",
    "@com_google_absl//absl/synchronization",
    "@com_google_absl//absl/types:optional",
    "//tensorflow/compiler/jit/ops:xla_ops",
    "//tensorflow/compiler/tf2xla:layout_util",
    "//tensorflow/compiler/tf2xla:common",
    "//tensorflow/compiler/tf2xla:tf2xla_util",
    "//tensorflow/compiler/tf2xla:xla_compiler",
    "//tensorflow/compiler/tf2xla:xla_op_registry",
    "//tensorflow/compiler/tf2xla/kernels:xla_dummy_ops",
    "//tensorflow/compiler/tf2xla/kernels:xla_ops",
    "@local_xla//xla:util",
    "@local_xla//xla/client:client_library",
    "@local_xla//xla/client:global_data",
    "@local_xla//xla/client:local_client",
    "@local_xla//xla/service:stream_pool",
    "@local_xla//xla/service/gpu:gpu_executable_run_options",
    "//tensorflow/core:array_ops_op_lib",
    "//tensorflow/core:control_flow_ops_op_lib",
    "//tensorflow/core:core_cpu",
    "//tensorflow/core:core_cpu_internal",
    "//tensorflow/core:dataset_ops_op_lib",
    "//tensorflow/core:framework",
    "//tensorflow/core:framework_internal",
    "//tensorflow/core:functional_ops_op_lib",
    "//tensorflow/core:lib",
    "//tensorflow/core:lib_internal",
    "//tensorflow/core:math_ops_op_lib",
    "//tensorflow/core:nn_ops_op_lib",
    "//tensorflow/core:no_op_op_lib",
    "//tensorflow/core:protos_all_cc",
    "//tensorflow/core:resource_variable_ops_op_lib",
    "//tensorflow/core:sendrecv_ops_op_lib",
    "//tensorflow/core:state_ops_op_lib",
    "//tensorflow/core/platform:stream_executor_no_cuda",
    "//tensorflow/core/kernels:constant_op",
    "//tensorflow/core/kernels:fifo_queue",
    "//tensorflow/core/kernels:function_ops",
    "//tensorflow/core/kernels:identity_op",
    "//tensorflow/core/kernels:resource_variable_ops",
    "//tensorflow/core/kernels:shape_ops",
    "//tensorflow/core/kernels:variable_ops",
    "//tensorflow/core/kernels/data:finalize_dataset_op",
    "//tensorflow/core/kernels/data:generator_dataset_op",
    "//tensorflow/core/kernels/data:iterator_ops",
    "//tensorflow/core/kernels/data:optional_ops",
    "//tensorflow/core/kernels/data:prefetch_dataset_op",
    "//tensorflow/core/kernels/data:options_dataset_op",
    "//tensorflow/core/profiler/lib:traceme",
    "//tensorflow/core/tfrt/common:async_value_tensor",
    "@local_xla//xla/stream_executor/integrations:tf_allocator_adapter",
    "@local_xla//xla/stream_executor/platform",
]

cc_library(
    name = "xla_device_context",
    srcs = ["xla_device_context.cc"],
    hdrs = ["xla_device_context.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_launch_util",
        ":xla_tensor",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_helpers",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core/common_runtime:device",
        "//tensorflow/core/common_runtime:dma_helper",
        "//tensorflow/core/framework:allocator",
        "@com_google_absl//absl/synchronization",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:util",
        "@local_xla//xla/client:global_data",
        "@local_xla//xla/client:local_client",
    ],
)

cc_library(
    name = "xla_device_no_jit_rewrite_registration",
    srcs = [
        "xla_compile_on_demand_op.cc",
        "xla_compiler_options_util.cc",
        "xla_device.cc",
        "xla_device_ops.cc",
        "xla_ops_on_regular_devices.cc",
        "xla_platform_info.cc",
    ],
    hdrs = [
        "xla_compile_on_demand_op.h",
        "xla_compiler_options_util.h",
        "xla_device.h",
        "xla_device_ops.h",
        "xla_platform_info.h",
    ],
    # Public visibility is needed for external TF/XLA backends.
    visibility = ["//visibility:public"],
    deps = XLA_DEVICE_DEPS + [
        ":device_compilation_cache",
        ":device_compilation_profiler",
        ":device_compiler",
        ":device_compiler_client",
        ":device_executable_persistor",
        ":flags_headers",
        ":pjrt_base_device",
        ":pjrt_device_compiler_client",
        ":xla_device_compiler_client",
        ":xla_device_context",
        "//tensorflow/core/tfrt/common:create_pjrt_client_util",
        "//tensorflow/core/tfrt/common:global_state",
        "//tensorflow/core/tfrt/common:pjrt_util",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/framework:device_id_utils",
        "@local_xla//xla:executable_run_options",
        "@local_xla//xla/hlo/ir:hlo",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt:tf_pjrt_client",
        "@local_xla//xla/service:compiler",
        "@local_xla//xla/service:executable",
        "@local_xla//xla/stream_executor:platform_manager",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xla_device",
    hdrs = [
        "xla_compile_on_demand_op.h",
        "xla_device.h",
        "xla_device_ops.h",
    ],
    # Public visibility is needed for external TF/XLA backends.
    visibility = ["//visibility:public"],
    deps = XLA_DEVICE_DEPS + [
        ":device_compilation_profiler",
        ":jit_compilation_passes",
        ":xla_device_no_jit_rewrite_registration",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "shape_inference_helpers",
    srcs = ["shape_inference_helpers.cc"],
    hdrs = ["shape_inference_helpers.h"],
    visibility = [":friends"],
    deps = select({
        "//tensorflow:android": [
            "//tensorflow/core:portable_tensorflow_lib",
        ],
        "//conditions:default": [
            "//tensorflow/core:graph",
        ],
    }),
)

cc_library(
    name = "flags",
    srcs = ["flags.cc"],
    hdrs = ["flags.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/compiler/mlir/tensorflow:dump_graph",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/tpu/kernels:sparse_core_xla_flags_defaults",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:parse_flags_from_env",
    ],
)

# Header-only version of "flags" library, for linking from the shared object
# without ODR violations.
cc_library(
    name = "flags_headers",
    hdrs = ["flags.h"],
    visibility = [":friends"],
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_header_only_library(
    name = "flags_headers_only",
    features = [
        "-parse_headers",  # buildifier: disable=no-parse-headers
    ],
    deps = [":flags_headers"],
)

cc_library(
    name = "common",
    srcs = [
        "defs.cc",
    ],
    hdrs = [
        "defs.h",
    ],
    visibility = [":friends"],
)

# Internal targets below this point.

cc_library(
    name = "variable_info",
    srcs = ["variable_info.cc"],
    hdrs = ["variable_info.h"],
    visibility = [
        ":internal",
        # We reuse VariableInfo in TFRT's implementation of TpuExecuteOp.
        "//learning/brain/tfrt/tf_tpu:__pkg__",
        "//learning/brain/tfrt/tpu_plugin:__pkg__",
        "//learning/brain/tfrt/tpu_common:__pkg__",
        "//tensorflow/core/common_runtime/next_pluggable_device:__pkg__",
    ],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "variable_info_util",
    srcs = ["variable_info_util.cc"],
    hdrs = ["variable_info_util.h"],
    visibility = [
        ":internal",
        # We reuse VariableInfo in TFRT's implementation of TpuExecuteOp.
        "//learning/brain/tfrt/tf_tpu:__pkg__",
        "//learning/brain/tfrt/tpu_plugin:__pkg__",
        "//learning/brain/tfrt/tpu_common:__pkg__",
        "//tensorflow/core/common_runtime/next_pluggable_device:__pkg__",
    ],
    deps = [
        ":variable_info",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/algorithm:container",
    ],
)

cc_library(
    name = "pjrt_compile_util",
    srcs = ["pjrt_compile_util.cc"],
    hdrs = ["pjrt_compile_util.h"],
    visibility = [
        ":internal",
        "//tensorflow/core/tfrt/gpu/kernel:__pkg__",
    ],
    deps = [
        ":device_compilation_profiler",
        ":device_compiler",
        ":xla_compile_util",
        ":xla_device_no_jit_rewrite_registration",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

tf_cuda_cc_test(
    name = "pjrt_compile_util_test",
    srcs = ["pjrt_compile_util_test.cc"],
    tags = [
        "config-cuda-only",
        "no_oss",  # This test only runs with GPU.
        "requires-gpu-nvidia",
        "xla",
    ],
    deps = [
        ":pjrt_compile_util",
        ":test_util",
        ":xla_gpu_jit",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:math_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/tfrt/common:pjrt_util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "xla_launch_util",
    srcs = ["xla_launch_util.cc"],
    hdrs = ["xla_launch_util.h"],
    visibility = [
        ":internal",
        # We reuse VariableInfo in TFRT's implementation of TpuExecuteOp.
        "//learning/brain/tfrt/tf_tpu:__pkg__",
        "//learning/brain/tfrt/tpu_plugin:__pkg__",
        "//learning/brain/tfrt/tpu_common:__pkg__",
        "//tensorflow/core/common_runtime/next_pluggable_device:__pkg__",
        "//tensorflow/core/tfrt/gpu/kernel:__pkg__",
    ],
    deps = [
        ":pjrt_tensor_buffer",
        ":pjrt_tensor_buffer_util",
        ":variable_info",
        ":variable_info_util",
        ":xla_tensor",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:dma_helper",
        "//tensorflow/core/common_runtime/gpu:gpu_serving_device_selector",
        "//tensorflow/core/tfrt/common:async_value_tensor",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@local_tsl//tsl/framework:device_id_utils",
        "@local_tsl//tsl/framework:serving_device_selector_policies",
        "@local_xla//xla:shape_util",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt:pjrt_future",
        "@local_xla//xla/pjrt:pjrt_stream_executor_client",
        "@local_xla//xla/pjrt:tracked_device_buffer",
        "@local_xla//xla/service:shaped_buffer",
        "@local_xla//xla/stream_executor:device_memory_allocator",
        "@local_xla//xla/stream_executor:platform_manager",
    ],
)

tf_cc_test(
    name = "xla_launch_util_test",
    srcs = ["xla_launch_util_test.cc"],
    deps = [
        ":device_compiler",
        ":flags_headers",
        ":pjrt_device_compiler_client",
        ":variable_info",
        ":variable_info_util",
        ":xla_cpu_device",
        ":xla_cpu_jit",
        ":xla_device_no_jit_rewrite_registration",
        ":xla_launch_util",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:fake_input",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/tfrt/common:create_pjrt_client_util",
        "//tensorflow/core/tfrt/common:pjrt_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/framework:device_id_utils",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt:tfrt_cpu_pjrt_client",
        "@local_xla//xla/tests:literal_test_util",
    ],
)

tf_cuda_only_cc_test(
    name = "xla_launch_util_gpu_test",
    srcs = ["xla_launch_util_gpu_test.cc"],
    deps = [
        ":device_compiler",
        ":flags_headers",
        ":pjrt_device_compiler_client",
        ":pjrt_device_context",
        ":variable_info",
        ":variable_info_util",
        ":xla_device_no_jit_rewrite_registration",
        ":xla_gpu_jit",
        ":xla_launch_util",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:fake_input",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/tfrt/common:create_pjrt_client_util",
        "//tensorflow/core/tfrt/common:pjrt_util",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/lib/core:status_test_util",
        "@local_tsl//tsl/platform:status",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/tests:literal_test_util",
    ],
)

cc_library(
    name = "xla_compile_util",
    srcs = ["xla_compile_util.cc"],
    hdrs = ["xla_compile_util.h"],
    visibility = [
        ":internal",
        "//tensorflow/core/common_runtime/next_pluggable_device:__pkg__",
        "//tensorflow/core/tfrt/gpu/kernel:__pkg__",
    ],
    deps = [
        ":flags_headers",
        "//tensorflow/compiler/tf2xla:xla_argument",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core/tfrt/common:global_state",
        "//tensorflow/core/util:determinism",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "xla_compile_util_test",
    srcs = [
        "xla_compile_util_test.cc",
    ],
    deps = [
        ":flags_headers",
        ":xla_compile_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:test_main",
        "//tensorflow/core/framework:fake_input",
        "//tensorflow/core/kernels:identity_op",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_googletest//:gtest",
    ],
)

tf_proto_library(
    name = "xla_compilation_cache_proto",
    srcs = ["xla_compilation_cache.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos() + ["@local_xla//xla/service:hlo_proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "device_compiler",
    hdrs = ["device_compiler.h"],
    copts = tf_copts(),
    visibility = [
        ":internal",
        "//tensorflow/core/common_runtime/next_pluggable_device:__pkg__",
    ],
    deps = [
        ":device_compilation_cache",
        ":device_compilation_cluster_signature",
        ":device_compilation_profiler",
        ":device_compiler_client",
        ":device_executable_persistor",
        ":flags_headers",
        ":tf_graph_to_hlo_compiler",
        ":xla_compile_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:thread_annotations",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/client:local_client",
    ],
)

tf_cc_test(
    name = "device_compiler_disable_test",
    srcs = [
        "device_compiler_disable_test.cc",
    ],
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":device_compilation_profiler",
        ":device_compiler",
        ":flags",
        ":xla_cpu_jit",
        ":xla_device_compiler_client",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@local_xla//xla/client:client_library",
    ],
)

cc_library(
    name = "jit_compilation_passes",
    srcs = ["jit_compilation_pass_registration.cc"],
    deps = [
        ":compilation_passes",
        ":xla_activity_logging_listener",
        "//tensorflow/compiler/tf2xla:functionalize_control_flow_pass_registration",
        "//tensorflow/compiler/tf2xla:mlir_bridge_pass_registration",
        "//tensorflow/core:core_cpu_internal",
    ],
    alwayslink = 1,
)

cc_library(
    name = "get_compiler_ir",
    srcs = ["get_compiler_ir.cc"],
    hdrs = ["get_compiler_ir.h"],
    visibility = [":internal"],
    deps = [
        ":compilability_check_util",
        ":device_compiler",
        ":variable_info",
        ":variable_info_util",
        ":xla_device_no_jit_rewrite_registration",
        ":xla_launch_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime:core_cpu_internal",
        "//tensorflow/core/common_runtime/eager:tensor_handle",
        "//tensorflow/core/platform:refcount",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:status_macros",
        "@local_xla//xla/client:executable_build_options",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/service:hlo_graph_dumper",
        "@local_xla//xla/stream_executor:platform",
        "@local_xla//xla/stream_executor/host:host_platform_id",
    ],
    alwayslink = 1,
)

# Header-only version of "flags" library, for linking from the shared object
# without ODR violations.
cc_library(
    name = "get_compiler_ir_hdrs",
    textual_hdrs = ["get_compiler_ir.h"],
    visibility = [":internal"],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)

cc_header_only_library(
    name = "get_compiler_ir_hdrs_only",
    features = [
        "-parse_headers",  # buildifier: disable=no-parse-headers
    ],
    deps = [":get_compiler_ir_hdrs"],
)

# This target can be used by XLA device plugins to prevent circular dependencies, and provides access to all of the required headers for building a device library.
cc_header_only_library(
    name = "xla_jit_headers_lib",
    visibility = ["//visibility:public"],
    deps = [
        ":xla_cpu_device",
        ":xla_cpu_jit",
        ":xla_gpu_device",
        ":xla_gpu_jit",
    ],
)

cc_library(
    name = "xla_kernel_creator",
    srcs = [
        "xla_kernel_creator.cc",
    ],
    hdrs = [
        "xla_kernel_creator.h",
    ],
    visibility = [
        ":internal",
        "//tensorflow/core/common_runtime/eager:__pkg__",
    ],
    deps = [
        ":common",
        ":compilability_check_util",
        ":compilation_passes",
        ":flags",
        ":jit_compilation_passes",
        "//tensorflow/compiler/jit/kernels:xla_ops_no_jit_rewrite_registration",
        "//tensorflow/compiler/tf2xla:mlir_bridge_pass",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "xla_kernel_creator_test",
    srcs = [
        "xla_kernel_creator.h",
        "xla_kernel_creator_test.cc",
    ],
    deps = [
        ":xla_kernel_creator",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "resource_operation_safety_analysis",
    srcs = ["resource_operation_safety_analysis.cc"],
    hdrs = ["resource_operation_safety_analysis.h"],
    deps = [
        ":xla_cluster_util",
        "//tensorflow/compiler/tf2xla:resource_operation_table",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla/service/graphcycles",
    ],
)

tf_cc_test(
    name = "resource_operation_safety_analysis_test",
    srcs = ["resource_operation_safety_analysis_test.cc"],
    deps = [
        ":common",
        ":resource_operation_safety_analysis",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "shape_inference",
    srcs = ["shape_inference.cc"],
    hdrs = ["shape_inference.h"],
    visibility = [":friends"],
    deps = [
        ":shape_inference_helpers",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:statusor",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        ":shape_inference",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "@local_xla//xla:status_macros",
    ],
)

tf_cc_test(
    name = "shape_inference_test",
    srcs = ["shape_inference_test.cc"],
    deps = [
        ":shape_inference",
        ":test_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/kernels:constant_op",
        "@local_tsl//tsl/platform:status",
    ],
)

cc_library(
    name = "encapsulate_util",
    srcs = ["encapsulate_util.cc"],
    hdrs = ["encapsulate_util.h"],
    deps = [
        ":shape_inference",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

tf_cc_test(
    name = "encapsulate_util_test",
    srcs = ["encapsulate_util_test.cc"],
    deps = [
        ":encapsulate_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "compilation_passes",
    srcs = [
        "build_xla_ops_pass.cc",
        "clone_constants_for_better_clustering.cc",
        "cluster_scoping_pass.cc",
        "deadness_analysis.cc",
        "deadness_analysis_internal.h",
        "encapsulate_subgraphs_pass.cc",
        "encapsulate_xla_computations_pass.cc",
        "extract_outside_compilation_pass.cc",
        "force_xla_constants_on_host_pass.cc",
        "increase_dynamism_for_auto_jit_pass.cc",
        "mark_for_compilation_pass.cc",
        "mark_for_compilation_pass_test_helper.cc",
        "partially_decluster_pass.cc",
        "report_clustering_info_pass.cc",
    ],
    hdrs = [
        "build_xla_ops_pass.h",
        "clone_constants_for_better_clustering.h",
        "cluster_scoping_pass.h",
        "deadness_analysis.h",
        "encapsulate_subgraphs_pass.h",
        "encapsulate_xla_computations_pass.h",
        "extract_outside_compilation_pass.h",
        "force_xla_constants_on_host_pass.h",
        "increase_dynamism_for_auto_jit_pass.h",
        "mark_for_compilation_pass.h",
        "mark_for_compilation_pass_test_helper.h",
        "partially_decluster_pass.h",
        "report_clustering_info_pass.h",
    ],
    visibility = [
        ":internal",
        "//tensorflow/core/tfrt/utils:__pkg__",
        "//third_party/cloud_tpu/inference_converter:__pkg__",
        "//waymo/onboard/ml/chauffeur_net:__pkg__",
    ],
    deps = [
        "compilability_check_util",
        ":common",
        ":device_util",
        ":encapsulate_util",
        ":flags",
        ":resource_operation_safety_analysis",
        ":shape_inference_helpers",
        ":xla_activity_listener",
        ":xla_cluster_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:scope_internal",
        "//tensorflow/compiler/jit/ops:xla_ops",
        "//tensorflow/compiler/tf2xla:resource_operation_table",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/compiler/tf2xla/cc:xla_jit_ops",
        "//tensorflow/compiler/tf2xla/cc:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:bounds_check",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:statusor",
        "@local_xla//xla:union_find",
        "@local_xla//xla:util",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/service/graphcycles",
    ],
)

cc_library(
    name = "xla_cluster_util",
    srcs = ["xla_cluster_util.cc"],
    hdrs = ["xla_cluster_util.h"],
    deps = [
        ":flags",
        ":xla_activity_proto_cc",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/framework:bounds_check",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:statusor",
        "@local_xla//xla/service/graphcycles",
    ],
)

cc_library(
    name = "device_util",
    srcs = ["device_util.cc"],
    hdrs = ["device_util.h"],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:framework",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/numeric:bits",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:status_macros",
        "@local_xla//xla:statusor",
    ],
)

tf_cc_test(
    name = "device_util_test",
    srcs = ["device_util_test.cc"],
    deps = [
        ":device_util",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

tf_cc_test(
    name = "deadness_analysis_test",
    size = "small",
    srcs = [
        "deadness_analysis_internal.h",
        "deadness_analysis_test.cc",
    ],
    deps = [
        ":common",
        ":compilation_passes",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

cc_library(
    name = "compilation_passes_test_main",
    testonly = True,
    srcs = ["compilation_passes_test_main.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":flags",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "compilation_passes_test",
    size = "small",
    srcs = [
        "build_xla_ops_pass_test.cc",
        "clone_constants_for_better_clustering_test.cc",
        "cluster_scoping_pass_test.cc",
        "encapsulate_subgraphs_pass_test.cc",
        "encapsulate_xla_computations_pass_test.cc",
        "extract_outside_compilation_pass_test.cc",
        "force_xla_constants_on_host_pass_test.cc",
        "increase_dynamism_for_auto_jit_pass_test.cc",
        "mark_for_compilation_pass_test.cc",
        "partially_decluster_pass_test.cc",
        "rearrange_function_argument_pass_test.cc",
    ],
    tags = [
        # TODO(b/141643254) Re-enable msan after fixing
        # use-of-uninitialized-value error.
        "nomsan",
    ] + tf_cuda_tests_tags(),
    deps = [
        ":common",
        ":compilability_check_util",
        ":compilation_passes",
        ":compilation_passes_test_main",
        ":encapsulate_util",
        ":flags",
        ":node_matchers",
        ":test_util",
        ":xla_cluster_util",
        ":xla_cpu_device",
        ":xla_gpu_device",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/cc:sendrecv_ops",
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla:rearrange_function_argument",
        "//tensorflow/compiler/tf2xla:side_effect_util",
        "//tensorflow/compiler/tf2xla:test_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/cc:xla_jit_ops",
        "//tensorflow/compiler/tf2xla/cc:xla_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_dummy_ops",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core:test",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:test",
    ],
)

tf_cc_test(
    name = "xla_cluster_util_test",
    size = "small",
    srcs = [
        "xla_cluster_util_test.cc",
    ],
    deps = [
        ":common",
        ":xla_cluster_util",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/compiler/jit/kernels:xla_ops",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/kernels:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@local_xla//xla:status_macros",
    ],
)

cc_library(
    name = "node_matchers",
    testonly = True,
    srcs = ["node_matchers.cc"],
    hdrs = ["node_matchers.h"],
    deps = [
        "//tensorflow/cc:ops",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla:test",
    ],
)

tf_cc_test(
    name = "node_matchers_test",
    srcs = ["node_matchers_test.cc"],
    deps = [
        ":node_matchers",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:cc_ops_internal",
        "//tensorflow/cc:ops",
        "//tensorflow/core:ops",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "compilability_check_util",
    srcs = ["compilability_check_util.cc"],
    hdrs = ["compilability_check_util.h"],
    visibility = [
        ":friends",
    ],
    deps = [
        ":common",
        ":device_util",
        ":flags",
        ":resource_operation_safety_analysis",
        ":xla_activity_listener",
        ":xla_activity_proto_cc",
        ":xla_cluster_util",
        "//tensorflow/compiler/tf2xla:resource_operation_table",
        "//tensorflow/compiler/tf2xla:tf2xla_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:graph",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_xla//xla:statusor",
        "@local_xla//xla:union_find",
        "@local_xla//xla:util",
        "@local_xla//xla/service/graphcycles",
    ],
)

tf_cc_test(
    name = "compilability_check_util_test",
    srcs = ["compilability_check_util_test.cc"],
    deps = [
        ":compilability_check_util",
        ":xla_cpu_device",
        ":xla_cpu_jit",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:functional_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/tf2xla:test_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla/cc:xla_jit_ops",
        "//tensorflow/compiler/tf2xla/cc:xla_ops",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/memory",
    ],
)

tf_cc_test(
    name = "xla_activity_listener_test",
    srcs = ["xla_activity_listener_test.cc"],
    deps = [
        ":flags",
        ":xla_activity_listener",
        ":xla_cpu_device",
        ":xla_cpu_jit",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:ops",
        "//tensorflow/core:all_kernels",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:ops",
        "//tensorflow/core:test",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/kernels:cwise_op",
        "//tensorflow/core/kernels:matmul_op",
        "//tensorflow/core/kernels:partitioned_function_ops",
    ],
)

tf_custom_op_py_strict_library(
    name = "xla_ops_py",
    kernels = ["//tensorflow/compiler/jit/ops:xla_ops"],
    visibility = [
        ":friends",
    ],
    deps = [
        "//tensorflow/compiler/jit/ops:xla_ops_grad",
        "//tensorflow/compiler/jit/ops:xla_ops_wrapper_py",
    ],
)

cc_library(
    name = "xla_activity_listener",
    srcs = ["xla_activity_listener.cc"],
    hdrs = ["xla_activity_listener.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":xla_activity_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_proto_library(
    name = "xla_activity_proto",
    srcs = ["xla_activity.proto"],
    cc_api_version = 2,
    protodeps = tf_additional_all_protos(),
)

cc_library(
    name = "xla_activity_logging_listener",
    srcs = ["xla_activity_logging_listener.cc"],
    deps = [
        ":xla_activity_listener",
        ":xla_activity_proto_cc",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,
)

cc_library(
    name = "tf_to_hlo_compiler",
    hdrs = ["tf_to_hlo_compiler.h"],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework",
    ],
)

cc_library(
    name = "tf_graph_to_hlo_compiler",
    srcs = ["tf_graph_to_hlo_compiler.cc"],
    hdrs = ["tf_graph_to_hlo_compiler.h"],
    deps = [
        ":tf_to_hlo_compiler",
        "//tensorflow/compiler/tf2xla:xla_compiler",
    ],
)

cc_library(
    name = "device_compilation_profiler",
    srcs = ["device_compilation_profiler.cc"],
    hdrs = ["device_compilation_profiler.h"],
    deps = [
        ":xla_activity_listener",
        ":xla_activity_proto_cc",
        ":xla_compile_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/platform:mutex",
    ],
)

cc_library(
    name = "device_compiler_client",
    srcs = ["device_compiler_client.cc"],
    hdrs = ["device_compiler_client.h"],
    visibility = [
        ":internal",
        "//tensorflow/core/common_runtime/next_pluggable_device:__pkg__",
    ],
    deps = [
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core/util:determinism",
        "@local_xla//xla/client:executable_build_options",
    ],
)

tf_cc_test(
    name = "device_compiler_client_test",
    srcs = ["device_compiler_client_test.cc"],
    deps = [
        ":device_compiler_client",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "xla_device_compiler_client",
    srcs = ["xla_device_compiler_client.cc"],
    hdrs = ["xla_device_compiler_client.h"],
    deps = [
        ":device_compiler_client",
        "@local_xla//xla/client:local_client",
    ],
)

cc_library(
    name = "device_executable_persistor",
    hdrs = ["device_executable_persistor.h"],
    deps = [
        ":xla_compilation_cache_proto_cc",
        ":xla_device_compiler_client",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib_headers_for_pybind",
        "//tensorflow/core/platform:path",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/log",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:util",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/service:hlo_proto_cc",
    ],
)

cc_library(
    name = "device_compilation_cache",
    hdrs = ["device_compilation_cache.h"],
    deps = [
        ":device_compilation_cluster_signature",
        ":xla_compile_util",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/core:framework_lite",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "device_compilation_cluster_signature",
    srcs = ["device_compilation_cluster_signature.cc"],
    hdrs = ["device_compilation_cluster_signature.h"],
    deps = ["//tensorflow/compiler/tf2xla:xla_compiler"],
)

cc_library(
    name = "pjrt_device_compiler_client",
    srcs = ["pjrt_device_compiler_client.cc"],
    hdrs = ["pjrt_device_compiler_client.h"],
    deps = [
        ":device_compiler_client",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "pjrt_base_device",
    srcs = ["pjrt_base_device.cc"],
    hdrs = ["pjrt_base_device.h"],
    # Public visibility is needed for external TF/XLA backends.
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:local_device",
    ],
)

cc_library(
    name = "pjrt_tensor_buffer",
    hdrs = ["pjrt_tensor_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "pjrt_tensor_buffer_util",
    srcs = ["pjrt_tensor_buffer_util.cc"],
    hdrs = ["pjrt_tensor_buffer_util.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_tensor_buffer",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:dma_helper",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

tf_cuda_only_cc_test(
    name = "pjrt_tensor_buffer_util_test",
    srcs = ["pjrt_tensor_buffer_util_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "config-cuda-only",
        "no_oss",
    ],
    deps = [
        ":pjrt_tensor_buffer_util",
        ":test_util",
        ":xla_gpu_jit",
        "//tensorflow/core:framework",
        "//tensorflow/core/tfrt/common:pjrt_util",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/platform:statusor",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

cc_library(
    name = "pjrt_device_context",
    srcs = [
        "pjrt_device_context.cc",
    ],
    hdrs = [
        "pjrt_device_context.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":pjrt_tensor_buffer",
        ":pjrt_tensor_buffer_util",
        "//tensorflow/c/experimental/next_pluggable_device:tensor_pjrt_buffer_util",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/core:framework",
        "//tensorflow/core/common_runtime:dma_helper",
        "//tensorflow/core/common_runtime/next_pluggable_device:next_pluggable_device_api",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/tfrt/common:async_value_tensor",
        "//tensorflow/core/tfrt/common:create_pjrt_client_util",
        "@com_google_absl//absl/status",
        "@local_tsl//tsl/framework:device_id_utils",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/tsl/c:tsl_status_internal",
    ],
)

cc_library(
    name = "xla_host_recv_device_context",
    srcs = [
        "xla_host_recv_device_context.cc",
    ],
    hdrs = [
        "xla_host_recv_device_context.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core:framework",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor:device_memory",
        "@tf_runtime//:async_value",
    ],
)

cc_library(
    name = "xla_host_send_device_context",
    srcs = [
        "xla_host_send_device_context.cc",
    ],
    hdrs = [
        "xla_host_send_device_context.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core:framework",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor:device_memory",
        "@tf_runtime//:async_value",
    ],
)

tf_cuda_only_cc_test(
    name = "xla_host_send_recv_device_context_test",
    srcs = ["xla_host_send_recv_device_context_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "config-cuda-only",
        "no_oss",  # Temporarily disable OSS.
    ],
    deps = [
        ":flags",
        ":xla_device",
        ":xla_gpu_device",
        ":xla_host_recv_device_context",
        ":xla_host_send_device_context",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:tensor_testutil",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla:shape_util",
        "@local_xla//xla/stream_executor",
        "@local_xla//xla/stream_executor:device_memory",
        "@local_xla//xla/stream_executor:platform_manager",
    ],
)

tf_cc_test(
    name = "device_compilation_cluster_signature_test",
    srcs = [
        "device_compilation_cluster_signature_test.cc",
    ],
    deps = [
        ":device_compilation_cluster_signature",
        ":flags",
        "//tensorflow/compiler/tf2xla:common",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@local_xla//xla/client:client_library",
    ],
)

tf_cc_test(
    name = "device_compilation_profiler_test",
    srcs = ["device_compilation_profiler_test.cc"],
    tags = [
        "nomsan",  # TODO(b/284492454)
    ],
    deps = [
        ":device_compilation_profiler",
        ":xla_activity_proto_cc",
        "//tensorflow/compiler/jit/tests:device_compiler_test_helper",
        "//tensorflow/core:protos_all_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "device_executable_persistor_test",
    srcs = ["device_executable_persistor_test.cc"],
    tags = ["no_cuda_on_cpu_tap"],
    deps = [
        ":device_compiler_client",
        ":device_executable_persistor",
        ":pjrt_device_compiler_client",
        ":xla_compilation_cache_proto_cc",
        ":xla_cpu_device",
        ":xla_cpu_jit",
        ":xla_device_compiler_client",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:math_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/tfrt/common:create_pjrt_client_util",
        "//tensorflow/core/tfrt/common:pjrt_util",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/client:executable_build_options",
        "@local_xla//xla/client:local_client",
        "@local_xla//xla/pjrt:pjrt_client",
        "@local_xla//xla/pjrt:tfrt_cpu_pjrt_client",
    ],
)

tf_cc_test(
    name = "device_compilation_cache_test",
    srcs = ["device_compilation_cache_test.cc"],
    deps = [
        ":device_compilation_cache",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:errors",
        "@com_google_googletest//:gtest_main",
        "@local_tsl//tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

# TODO(b/261212343): Support running this test on CPU and OSS.
tf_cuda_cc_test(
    name = "device_compiler_test",
    srcs = ["device_compiler_test.cc"],
    tags = [
        "config-cuda-only",
        "no_oss",  # This test only runs with GPU.
        "requires-gpu-nvidia",
        "xla",
    ],
    deps = [
        ":device_compilation_cluster_signature",
        ":device_compiler",
        ":device_compiler_client",
        ":xla_device_compiler_client",
        ":xla_gpu_device",
        ":xla_gpu_jit",
        "//tensorflow/cc:function_ops",
        "//tensorflow/cc:math_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/compiler/jit/tests:device_compiler_test_helper",
        "//tensorflow/core:framework",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:fake_input",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:notification",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/stream_executor:platform_manager",
    ],
)

tf_cuda_cc_test(
    name = "device_context_test",
    srcs = ["device_context_test.cc"],
    tags = tf_cuda_tests_tags() + [
        "config-cuda-only",
        "no_oss",  # Temporarily disable OSS.
    ],
    deps = [
        ":flags",
        ":xla_device",
        ":xla_gpu_device",
        ":xla_gpu_jit",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:test",
        "//tensorflow/core/framework:tensor_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cuda_cc_test(
    name = "xla_compiler_options_util_test",
    srcs = ["xla_compiler_options_util_test.cc"],
    tags = tf_cuda_tests_tags(),
    deps = [
        ":flags",
        ":pjrt_device_compiler_client",
        ":test_util",
        ":xla_device_no_jit_rewrite_registration",
        ":xla_gpu_device",
        ":xla_gpu_jit",
        "//tensorflow/compiler/tf2xla:xla_compiler",
        "//tensorflow/compiler/tf2xla:xla_op_registry",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla:xla_data_proto_cc",
        "@local_xla//xla/client:client_library",
        "@local_xla//xla/pjrt:pjrt_client",
    ],
)

tf_cuda_cc_test(
    name = "xla_platform_info_test",
    srcs = ["xla_platform_info_test.cc"],
    tags = tf_cuda_tests_tags() + ["config-cuda-only"],
    deps = [
        ":flags_headers",
        ":test_util",
        ":xla_device_no_jit_rewrite_registration",
        ":xla_gpu_device",
        ":xla_gpu_jit",
        "//tensorflow/compiler/tf2xla:layout_util",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:refcount",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:error_codes_proto_impl_cc",
        "//tensorflow/core/tfrt/common:create_pjrt_client_util",
        "//tensorflow/core/tfrt/common:pjrt_util",
        "//tensorflow/core/tpu:tpu_defs",
        "@com_google_googletest//:gtest_main",
        "@local_xla//xla/pjrt:tfrt_cpu_pjrt_client",
    ],
)
