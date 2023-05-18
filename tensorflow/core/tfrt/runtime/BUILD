load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # Authorized users go here.
        "//tensorflow/compiler/xla/service/gpu/...",
        "//tensorflow/core/tfrt/...",
        "//tensorflow/core/runtime_fallback/...",
        # copybara:uncomment "//tensorflow_serving/...",
        # copybara:uncomment "//learning/brain/experimental/tfrt/...",
        # copybara:uncomment "//learning/brain/tfrt/...",
        # copybara:uncomment "//learning/infra/mira/...",
    ],
)

cc_library(
    name = "runtime",
    srcs = ["runtime.cc"],
    hdrs = ["runtime.h"],
    deps = [
        ":work_queue_interface",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "//tensorflow/core/tfrt/graph_executor:graph_execution_options",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/cpu:core_runtime",
        "@tf_runtime//backends/cpu:core_runtime_alwayslink",
    ],
)

cc_library(
    name = "work_queue_interface",
    srcs = ["work_queue_interface.cc"],
    hdrs = ["work_queue_interface.h"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:threadpool_interface",
        "//tensorflow/core/profiler/lib:connected_traceme",
        "//tensorflow/core/profiler/lib:traceme_encode",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tf_cc_test(
    name = "work_queue_interface_test",
    srcs = ["work_queue_interface_test.cc"],
    deps = [
        ":work_queue_interface",
        "//tensorflow/core:test",
        "//tensorflow/core/tfrt/utils:thread_pool",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//cpp_tests:common",
    ],
)

cc_library(
    name = "tf_threadpool_concurrent_work_queue",
    srcs = ["tf_threadpool_concurrent_work_queue.cc"],
    hdrs = ["tf_threadpool_concurrent_work_queue.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":work_queue_interface",
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:threadpool_interface",
        "//tensorflow/core/tfrt/utils:thread_pool",
        "@llvm-project//llvm:Support",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tf_cc_test(
    name = "tf_threadpool_concurrent_work_queue_test",
    srcs = ["tf_threadpool_concurrent_work_queue_test.cc"],
    deps = [
        ":tf_threadpool_concurrent_work_queue",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:status_matchers",
        "//tensorflow/core/tfrt/utils:thread_pool",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)
