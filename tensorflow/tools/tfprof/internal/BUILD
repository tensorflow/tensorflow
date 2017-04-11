package(
    default_visibility = ["//tensorflow:__subpackages__"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "tfprof_stats",
    srcs = ["tfprof_stats.cc"],
    hdrs = ["tfprof_stats.h"],
    deps = [
        ":tfprof_graph",
        ":tfprof_node",
        ":tfprof_options",
        ":tfprof_scope",
        ":tfprof_show",
        ":tfprof_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/tools/tfprof:protos_all_cc",
    ],
)

cc_library(
    name = "tfprof_node",
    srcs = ["tfprof_node.cc"],
    hdrs = ["tfprof_node.h"],
    deps = [
        ":tfprof_options",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "tfprof_scope",
    srcs = ["tfprof_scope.cc"],
    hdrs = ["tfprof_scope.h"],
    deps = [
        ":tfprof_constants",
        ":tfprof_node",
        ":tfprof_options",
        ":tfprof_show",
        ":tfprof_tensor",
        ":tfprof_utils",
        "//tensorflow/c:c_api",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/tools/tfprof:protos_all_cc",
    ],
)

cc_library(
    name = "tfprof_graph",
    srcs = ["tfprof_graph.cc"],
    hdrs = ["tfprof_graph.h"],
    deps = [
        ":tfprof_constants",
        ":tfprof_node",
        ":tfprof_options",
        ":tfprof_show",
        ":tfprof_tensor",
        ":tfprof_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/tools/tfprof:protos_all_cc",
    ],
)

cc_library(
    name = "tfprof_show",
    srcs = ["tfprof_show.cc"],
    hdrs = ["tfprof_show.h"],
    deps = [
        ":tfprof_constants",
        ":tfprof_node",
        ":tfprof_options",
        ":tfprof_tensor",
        ":tfprof_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
        "//tensorflow/tools/tfprof:protos_all_cc",
    ],
)

tf_cc_test(
    name = "tfprof_show_test",
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
        ":tfprof_options",
        ":tfprof_stats",
        ":tfprof_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/tools/tfprof:protos_all_cc",
    ],
)

cc_library(
    name = "tfprof_utils",
    srcs = ["tfprof_utils.cc"],
    hdrs = ["tfprof_utils.h"],
    copts = ["-Wno-sign-compare"],
    deps = [
        ":tfprof_options",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:regexp_internal",
    ],
)

cc_library(
    name = "tfprof_options",
    srcs = ["tfprof_options.cc"],
    hdrs = ["tfprof_options.h"],
    deps = [
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:lib",
        "//tensorflow/tools/tfprof:protos_all_cc",
    ],
)

cc_library(
    name = "print_model_analysis_hdr",
    hdrs = [
        "print_model_analysis.h",
    ],
    deps = [
        "//tensorflow/core:framework_lite",
        "//tensorflow/core:protos_all_cc",
    ],
)

cc_library(
    name = "print_model_analysis",
    srcs = ["print_model_analysis.cc"],
    hdrs = ["print_model_analysis.h"],
    deps = [
        ":tfprof_options",
        ":tfprof_stats",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/tools/tfprof:protos_all_cc",
    ],
    alwayslink = 1,
)

tf_cc_test(
    name = "tfprof_stats_test",
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
        ":tfprof_options",
        ":tfprof_stats",
        ":tfprof_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/tools/tfprof:protos_all_cc",
    ],
)

cc_library(
    name = "tfprof_tensor",
    srcs = ["tfprof_tensor.cc"],
    hdrs = ["tfprof_tensor.h"],
    copts = ["-Wno-sign-compare"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/tools/tfprof:protos_all_cc",
    ],
)

tf_cc_test(
    name = "tfprof_tensor_test",
    srcs = ["tfprof_tensor_test.cc"],
    data = [
        "testdata/ckpt.data-00000-of-00001",
        "testdata/ckpt.index",
        "testdata/ckpt.meta",
        "testdata/graph.pbtxt",
    ],
    deps = [
        ":tfprof_options",
        ":tfprof_stats",
        ":tfprof_utils",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/tools/tfprof:protos_all_cc",
    ],
)

cc_library(
    name = "tfprof_constants",
    hdrs = ["tfprof_constants.h"],
    deps = [
    ],
)
# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
