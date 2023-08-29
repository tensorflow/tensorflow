load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_shared_test", "tf_cc_test")

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
        "//tensorflow/compiler/mlir/tfrt/...",
        "//tensorflow/core/tfrt/...",
        "//tensorflow/core/runtime_fallback/...",
        # copybara:uncomment "//tensorflow_serving/...",
        # copybara:uncomment "//learning/brain/experimental/tfrt/...",
        # copybara:uncomment "//learning/brain/tfrt/...",
        # copybara:uncomment "//learning/infra/mira/...",
        # copybara:uncomment "//learning/pathways/serving/...",
        # copybara:uncomment "//learning/serving/...",
        # copybara:uncomment "//quality/webanswers/servo2/...",
    ],
)

cc_library(
    name = "runtime",
    srcs = ["runtime.cc"],
    hdrs = ["runtime.h"],
    deps = [
        ":work_queue_interface",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "//tensorflow/core/runtime_fallback/kernel:kernel_fallback_tensor",
        "//tensorflow/core/tfrt/graph_executor:graph_execution_options",
        "@tf_runtime//:core_runtime_alwayslink",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:tensor",
        "@tf_runtime//backends/cpu:core_runtime",
        "@tf_runtime//backends/cpu:core_runtime_alwayslink",
    ],
)

tf_cc_shared_test(
    name = "runtime_test",
    srcs = ["runtime_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":runtime",
        "//tensorflow/c:tf_tensor",
        "//tensorflow/core:test",
        "@com_google_googletest//:gtest_main",
        "@tf_runtime//cpp_tests:common",
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

cc_library(
    name = "stream",
    srcs = ["stream.cc"],
    hdrs = ["stream.h"],
    deps = [
        ":channel",
        "//tensorflow/compiler/mlir/tensorflow",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/framework:tensor_proto_cc",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:random",
        "//tensorflow/tsl/profiler/lib:traceme",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/utility",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "channel",
    hdrs = ["channel.h"],
    deps = [
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

tf_cc_shared_test(
    name = "stream_test",
    srcs = ["stream_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":stream",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/framework:tensor_testutil",
        "//tensorflow/core/tfrt/saved_model:saved_model_testutil",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:statusor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_googletest//:gtest",
        "@tf_runtime//:hostcontext",
        "@tf_runtime//:support",
    ],
)

tf_cc_test(
    name = "channel_test",
    srcs = ["channel_test.cc"],
    deps = [
        ":channel",
        "//tensorflow/tsl/platform:env",
        "//tensorflow/tsl/platform:status_matchers",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)
