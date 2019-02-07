package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow:tensorflow.bzl", "if_not_windows")
load("//tensorflow:tensorflow.bzl", "tf_cuda_library")
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")

cc_library(
    name = "tfprof_stats",
    srcs = ["tfprof_stats.cc"],
    hdrs = ["tfprof_stats.h"],
    deps = [
        ":tfprof_code",
        ":tfprof_graph",
        ":tfprof_node",
        ":tfprof_op",
        ":tfprof_scope",
        ":tfprof_show",
        ":tfprof_timeline",
        ":tfprof_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
    ],
)

cc_library(
    name = "tfprof_timeline",
    srcs = ["tfprof_timeline.cc"],
    hdrs = ["tfprof_timeline.h"],
    deps = [
        ":tfprof_node_show",
        ":tfprof_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@jsoncpp_git//:jsoncpp",
    ],
)

cc_library(
    name = "tfprof_node",
    srcs = ["tfprof_node.cc"],
    hdrs = ["tfprof_node.h"],
    deps = [
        ":tfprof_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
    ],
)

cc_library(
    name = "tfprof_scope",
    srcs = ["tfprof_scope.cc"],
    hdrs = ["tfprof_scope.h"],
    deps = [
        ":tfprof_constants",
        ":tfprof_node",
        ":tfprof_show",
        ":tfprof_tensor",
        ":tfprof_utils",
        "//tensorflow/c:c_api",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
    ],
)

cc_library(
    name = "tfprof_op",
    srcs = ["tfprof_op.cc"],
    hdrs = ["tfprof_op.h"],
    deps = [
        ":tfprof_constants",
        ":tfprof_node",
        ":tfprof_show_multi",
        ":tfprof_tensor",
        ":tfprof_utils",
        "//tensorflow/c:c_api",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
    ],
)

cc_library(
    name = "tfprof_code",
    srcs = ["tfprof_code.cc"],
    hdrs = ["tfprof_code.h"],
    deps = [
        ":tfprof_constants",
        ":tfprof_node",
        ":tfprof_show_multi",
        ":tfprof_timeline",
        ":tfprof_utils",
        "//tensorflow/c:c_api",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
    ],
)

cc_library(
    name = "tfprof_graph",
    srcs = ["tfprof_graph.cc"],
    hdrs = ["tfprof_graph.h"],
    deps = [
        ":tfprof_constants",
        ":tfprof_node",
        ":tfprof_show",
        ":tfprof_tensor",
        ":tfprof_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
    ],
)

cc_library(
    name = "tfprof_node_show",
    srcs = ["tfprof_node_show.cc"],
    hdrs = ["tfprof_node_show.h"],
    deps = [
        ":tfprof_constants",
        ":tfprof_node",
        ":tfprof_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
    ],
)

cc_library(
    name = "tfprof_show",
    srcs = ["tfprof_show.cc"],
    hdrs = ["tfprof_show.h"],
    deps = [
        ":tfprof_constants",
        ":tfprof_node",
        ":tfprof_node_show",
        ":tfprof_tensor",
        ":tfprof_timeline",
        ":tfprof_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
    ],
)

cc_library(
    name = "tfprof_show_multi",
    srcs = ["tfprof_show_multi.cc"],
    hdrs = ["tfprof_show_multi.h"],
    deps = [
        ":tfprof_constants",
        ":tfprof_node",
        ":tfprof_node_show",
        ":tfprof_scope",
        ":tfprof_show",
        ":tfprof_tensor",
        ":tfprof_timeline",
        ":tfprof_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
    ],
)

tf_cc_test(
    name = "tfprof_show_test",
    size = "small",
    srcs = ["tfprof_show_test.cc"],
    data = [
        "testdata/ckpt.data-00000-of-00001",
        "testdata/ckpt.index",
        "testdata/ckpt.meta",
        "testdata/graph.pbtxt",
        "testdata/run_meta",
        "testdata/tfprof_log",
    ],
    deps = [
        ":tfprof_constants",
        ":tfprof_stats",
        ":tfprof_tf_testlib",
        ":tfprof_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
    ],
)

tf_cc_test(
    name = "tfprof_timeline_test",
    size = "small",
    srcs = ["tfprof_timeline_test.cc"],
    data = [
        "testdata/graph.pbtxt",
        "testdata/run_meta",
    ],
    deps = [
        ":tfprof_constants",
        ":tfprof_stats",
        ":tfprof_tf_testlib",
        ":tfprof_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
    ],
)

cc_library(
    name = "tfprof_utils",
    srcs = ["tfprof_utils.cc"],
    hdrs = ["tfprof_utils.h"],
    copts = if_not_windows(["-Wno-sign-compare"]),
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/core/profiler:tfprof_options",
    ],
)

cc_library(
    name = "print_model_analysis_hdr",
    hdrs = ["print_model_analysis.h"],
    deps = ["//tensorflow/core:framework"],
)

cc_library(
    name = "print_model_analysis",
    srcs = ["print_model_analysis.cc"],
    hdrs = ["print_model_analysis.h"],
    deps = [
        ":tfprof_stats",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
        "//tensorflow/core/profiler/internal/advisor:tfprof_advisor",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "tfprof_stats_test",
    size = "small",
    srcs = ["tfprof_stats_test.cc"],
    data = [
        "testdata/ckpt.data-00000-of-00001",
        "testdata/ckpt.index",
        "testdata/ckpt.meta",
        "testdata/graph.pbtxt",
        "testdata/run_meta",
        "testdata/tfprof_log",
    ],
    deps = [
        ":tfprof_constants",
        ":tfprof_stats",
        ":tfprof_tf_testlib",
        ":tfprof_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
    ],
)

cc_library(
    name = "tfprof_tensor",
    srcs = ["tfprof_tensor.cc"],
    hdrs = ["tfprof_tensor.h"],
    copts = if_not_windows(["-Wno-sign-compare"]),
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:protos_all_cc",
    ],
)

tf_cc_test(
    name = "tfprof_tensor_test",
    size = "small",
    srcs = ["tfprof_tensor_test.cc"],
    data = [
        "testdata/ckpt.data-00000-of-00001",
        "testdata/ckpt.index",
        "testdata/ckpt.meta",
        "testdata/graph.pbtxt",
    ],
    deps = [
        ":tfprof_stats",
        ":tfprof_tf_testlib",
        ":tfprof_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/profiler:protos_all_cc",
        "//tensorflow/core/profiler:tfprof_options",
    ],
)

cc_library(
    name = "tfprof_constants",
    hdrs = ["tfprof_constants.h"],
)

cc_library(
    name = "tfprof_tf_testlib",
    testonly = 1,
    deps = [
        ":tfprof_tf_lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)

cc_library(
    name = "tfprof_tf_lib",
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
    ],
)

tf_cuda_library(
    name = "traceme_recorder",
    srcs = ["traceme_recorder.cc"],
    hdrs = ["traceme_recorder.h"],
    visibility = [
        "//learning/brain/runtime:__pkg__",  # xprof_bridge
        "//perftools/accelerators/xprof/xprofilez:__pkg__",  # alias xprof::TraceMeRecorder
        "//tensorflow/core/profiler/internal/cpu:__pkg__",  # host_tracer
        "//tensorflow/core/profiler/lib:__pkg__",  # traceme
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/stream_executor/lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tf_cuda_cc_test(
    name = "traceme_recorder_test",
    srcs = ["traceme_recorder_test.cc"],
    deps = [
        ":traceme_recorder",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cuda_library(
    name = "profiler_interface",
    hdrs = [
        "profiler_interface.h",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)
