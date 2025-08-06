load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs", "tf_gen_op_wrapper_cc", "tf_gen_op_wrapper_py", "tf_kernel_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "preemption_sync_manager",
    hdrs = ["preemption_sync_manager.h"],
    deps = [
        "@local_xla//xla/tsl/distributed_runtime/preemption:preemption_sync_manager",
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
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_xla//xla/tsl/distributed_runtime/coordination:coordination_service_agent",
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
    extra_py_deps = [
        "//tensorflow/python:pywrap_tfe",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:tf_export",
    ],
    py_lib_rule = py_strict_library,
    deps = [":check_preemption_op_op_lib"],
)
