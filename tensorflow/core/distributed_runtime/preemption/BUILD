load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "preemption_sync_manager",
    hdrs = ["preemption_sync_manager.h"],
    deps = [
        "//tensorflow/core/distributed_runtime/coordination:coordination_service_agent",
        "//tensorflow/core/platform:status",
    ],
)

cc_library(
    name = "preemption_notifier",
    srcs = ["preemption_notifier.cc"],
    hdrs = ["preemption_notifier.h"],
    deps = [
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:mutex",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "preemption_notifier_test",
    srcs = ["preemption_notifier_test.cc"],
    deps = [
        ":preemption_notifier",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform:env",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)
