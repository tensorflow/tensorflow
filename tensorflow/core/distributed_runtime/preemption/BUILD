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
    hdrs = ["preemption_notifier.h"],
    deps = [
        "@com_google_absl//absl/time",
    ],
)
