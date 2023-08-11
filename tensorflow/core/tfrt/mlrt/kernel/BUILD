load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_shared_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        # copybara:uncomment "//learning/brain/experimental/tfrt:__subpackages__",
        # copybara:uncomment "//learning/brain/tfrt:__subpackages__",
        # copybara:uncomment "//learning/infra/mira/distributed:__subpackages__",
        "//tensorflow/core/tfrt/graph_executor:__subpackages__",
        "//tensorflow/core/tfrt/mlrt/application/tensorflow/tests:__subpackages__",
        "//tensorflow/core/tfrt/saved_model:__subpackages__",
        "//tensorflow/core/tfrt/tfrt_session:__subpackages__",
    ],
)

cc_library(
    name = "kernel",
    srcs = ["kernel.cc"],
    hdrs = ["kernel.h"],
    deps = [
        ":context",
        ":kernel_runner_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_utils",
        "//tensorflow/core/tfrt/fallback:device_with_custom_allocator",
        "//tensorflow/core/tfrt/mlrt/interpreter:async_handle",
        "//tensorflow/core/tfrt/mlrt/interpreter:attribute_span",
        "//tensorflow/core/tfrt/mlrt/interpreter:builtin_kernels",
        "//tensorflow/core/tfrt/mlrt/interpreter:context",
        "//tensorflow/core/tfrt/mlrt/interpreter:execute",
        "//tensorflow/core/tfrt/mlrt/interpreter:future",
        "//tensorflow/core/tfrt/mlrt/interpreter:register_span",
        "//tensorflow/core/tfrt/mlrt/interpreter:value",
        "//tensorflow/core/tfrt/utils",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/profiler/lib:traceme",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf_headers",
        "@tf_runtime//:async_value",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "batch_kernel",
    srcs = ["batch_kernel.cc"],
    hdrs = ["batch_kernel.h"],
    deps = [
        ":context",
        ":kernel_runner_utils",
        "//tensorflow/core:framework",
        "//tensorflow/core/kernels/batching_util:batch_scheduler_hdrs",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/runtime_fallback/runtime:fallback_batch_kernel",
        "//tensorflow/core/tfrt/fallback:op_kernel_runner_cache",
        "//tensorflow/core/tfrt/mlrt/interpreter:context",
        "//tensorflow/core/tfrt/mlrt/interpreter:execute",
        "//tensorflow/core/tfrt/utils:fallback_tensor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_protobuf//:protobuf_headers",
        "@tf_runtime//:async_value",
        "@tf_runtime//:hostcontext",
    ],
)

cc_library(
    name = "kernel_runner_utils",
    srcs = ["kernel_runner_utils.cc"],
    hdrs = ["kernel_runner_utils.h"],
    deps = [
        ":context",
        "//tensorflow/core:framework",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_utils",
        "//tensorflow/core/tfrt/fallback:op_kernel_runner",
        "//tensorflow/core/tfrt/mlrt/interpreter:context",
        "//tensorflow/core/tfrt/mlrt/interpreter:future",
        "//tensorflow/core/tfrt/mlrt/interpreter:register_span",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
    ],
)

cc_library(
    name = "context",
    hdrs = ["context.h"],
    deps = [
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_compat_request_state",
        "//tensorflow/core/tfrt/fallback:op_kernel_runner",
        "//tensorflow/core/tfrt/mlrt/interpreter:context",
        "@tf_runtime//:hostcontext",
    ],
)

tf_cc_shared_test(
    name = "kernel_test",
    srcs = ["kernel_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":batch_kernel",
        ":context",
        ":kernel",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/kernels:math",
        "//tensorflow/core/ops:math_ops_op_lib",
        "//tensorflow/core/tfrt/fallback:device_with_custom_allocator",
        "//tensorflow/core/tfrt/fallback:fallback_state",
        "//tensorflow/core/tfrt/mlrt/bytecode:executable",
        "//tensorflow/core/tfrt/mlrt/interpreter:execute",
        "//tensorflow/core/tfrt/mlrt/interpreter:future",
        "//tensorflow/core/tfrt/mlrt/interpreter:interpreter_testutil",
        "//tensorflow/tsl/lib/core:status_test_util",
        "//tensorflow/tsl/platform:status_matchers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:ref_count",
    ],
)
