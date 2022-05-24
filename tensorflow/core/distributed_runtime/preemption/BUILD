load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "tf_grpc_cc_dependencies")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "preemption_sync_manager",
    srcs = ["preemption_sync_manager.cc"],
    hdrs = ["preemption_sync_manager.h"],
    deps = [
        ":preemption_notifier",
        "//tensorflow/core/distributed_runtime/coordination:coordination_service_agent",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "preemption_notifier",
    srcs = ["preemption_notifier.cc"],
    hdrs = ["preemption_notifier.h"],
    deps = [
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "preemption_sync_manager_test",
    srcs = ["preemption_sync_manager_test.cc"],
    deps = [
        ":preemption_notifier",
        ":preemption_sync_manager",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/distributed_runtime/coordination:coordination_client",
        "//tensorflow/core/distributed_runtime/coordination:coordination_service",
        "//tensorflow/core/distributed_runtime/coordination:coordination_service_impl",
        "//tensorflow/core/distributed_runtime/coordination:coordination_service_agent",
        "//tensorflow/core/distributed_runtime/rpc:async_service_interface",
        "//tensorflow/core/distributed_runtime/rpc/coordination:grpc_coordination_client",
        "//tensorflow/core/distributed_runtime/rpc/coordination:grpc_coordination_service_impl",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/protobuf:for_core_protos_cc",
    ] + tf_grpc_cc_dependencies(),
)

tf_cc_test(
    name = "preemption_notifier_test",
    size = "small",
    srcs = ["preemption_notifier_test.cc"],
    deps = [
        ":preemption_notifier",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)
