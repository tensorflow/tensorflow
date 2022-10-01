load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_gen_op_libs", "tf_gen_op_wrapper_cc", "tf_gen_op_wrapper_py", "tf_kernel_library")
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
        "//tensorflow/compiler/xla/pjrt/distributed:client",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/distributed_runtime/coordination:coordination_service_agent",
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "//tensorflow/core/protobuf:coordination_service_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "preemption_notifier",
    srcs = ["preemption_notifier.cc"],
    hdrs = ["preemption_notifier.h"],
    # copybara:uncomment compatible_with = ["//buildenv/target:gce"],
    deps = [
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tf_cc_test(
    name = "preemption_sync_manager_test",
    size = "small",
    srcs = ["preemption_sync_manager_test.cc"],
    deps = [
        ":preemption_notifier",
        ":preemption_sync_manager",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
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
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:statusor",
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
        "//tensorflow/core/platform",
        "//tensorflow/core/platform:env",
        "//tensorflow/core/platform:errors",
        "//tensorflow/core/platform:mutex",
        "//tensorflow/core/platform:status",
        "//tensorflow/core/platform:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

cc_library(
    name = "check_preemption_op",
    srcs = ["check_preemption_op.cc"],
    deps = ["//tensorflow/core:framework"],
    alwayslink = 1,
)

tf_kernel_library(
    name = "check_preemption_op_kernel",
    srcs = ["check_preemption_op_kernel.cc"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:error_payloads",
        "//tensorflow/core/distributed_runtime/coordination:coordination_service_agent",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = ["check_preemption_op"],
    sub_directory = "",
    deps = ["//tensorflow/core:lib"],
)

tf_gen_op_wrapper_cc(
    name = "gen_cc_check_preemption_op",
    out_ops_file = "cc/check_preemption_op",
    deps = [":check_preemption_op_op_lib"],
)

cc_library(
    name = "check_preemption_ops_cc",
    srcs = ["cc/check_preemption_op.cc"],
    hdrs = ["cc/check_preemption_op.h"],
    deps = [
        ":check_preemption_op_op_lib",
        "//tensorflow/cc:const_op",
        "//tensorflow/cc:ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_check_preemption_op",
    out = "gen_check_preemption_op.py",
    deps = [":check_preemption_op_op_lib"],
)
