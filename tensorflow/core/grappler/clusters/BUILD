load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load("//tensorflow:tensorflow.bzl", "tf_cc_test", "tf_cuda_library")
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load(
    "//tensorflow/core/platform:build_config_root.bzl",
    "tf_cuda_tests_tags",
)
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

tf_cuda_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    cuda_deps = [
        "@local_config_cuda//cuda:cudnn_header",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/common_runtime/gpu:gpu_id",
        "@eigen_archive//:eigen3",
    ],
)

tf_cc_test(
    name = "utils_test",
    srcs = ["utils_test.cc"],
    linkstatic = if_cuda(1, 0),
    tags = tf_cuda_tests_tags(),
    deps = [
        ":utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/common_runtime/gpu:gpu_id",
    ],
)

filegroup(
    name = "pywrap_required_hdrs",
    srcs = [
        "cluster.h",
        "single_machine.h",
        "utils.h",
        "virtual_cluster.h",
    ],
    visibility = [
        "//tensorflow/python/grappler:__pkg__",
    ],
)

cc_library(
    name = "cluster",
    srcs = ["cluster.cc"],
    hdrs = [
        "cluster.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:session_options",
        "//tensorflow/core/grappler:grappler_item",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "virtual_cluster",
    srcs = ["virtual_cluster.cc"],
    hdrs = [
        "virtual_cluster.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":utils",
        "//tensorflow/core:core_cpu_base",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/grappler/costs:analytical_cost_estimator",
        "//tensorflow/core/grappler/costs:op_level_cost_estimator",
        "//tensorflow/core/grappler/costs:virtual_scheduler",
        "@com_google_absl//absl/status",
    ],
)

tf_cc_test(
    name = "virtual_cluster_test",
    srcs = ["virtual_cluster_test.cc"],
    deps = [
        ":virtual_cluster",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@local_xla//xla/tsl/protobuf:error_codes_proto_impl_cc",
    ],
)

cc_library(
    name = "single_machine",
    srcs = ["single_machine.cc"],
    hdrs = [
        "single_machine.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cluster",
        ":utils",
        "//tensorflow/cc:coordinator",
        "//tensorflow/cc:queue_runner",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/common_runtime:core_cpu",
        "//tensorflow/core/common_runtime:core_cpu_lib",
        "//tensorflow/core/common_runtime:direct_session_internal",
        "//tensorflow/core/common_runtime/gpu:gpu_id",
        "//tensorflow/core/framework:cost_graph_proto_cc",
        "//tensorflow/core/framework:graph_proto_cc",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/kernels:ops_util",
        "//tensorflow/core/protobuf:for_core_protos_cc",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "single_machine_test",
    srcs = ["single_machine_test.cc"],
    args = ["--heap_check="],  # The GPU tracer leaks memory. TODO(b/185483595): use a dependency instead of a flag
    tags = [
        "no_cuda_on_cpu_tap",
        "no_gpu",
        "nomsan",  # TODO(b/160921160): broken by NOAUTOROLLBACK CL
    ],
    deps = [
        ":single_machine",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:resource_variable_ops",
        "//tensorflow/cc:scope",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib_proto_parsing",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/grappler:grappler_item",
        "//tensorflow/core/grappler:utils",
        "//tensorflow/core/grappler/inputs:trivial_test_graph_input_yielder",
    ],
)
