licenses(["notice"])  # 3-Clause BSD

exports_files(["LICENSE"])

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

cc_library(
    name = "ngraph_tf",
    srcs = [
        "src/ngraph_api.h",
        "src/ngraph_api.cc",
        "src/ngraph_assign_clusters.h",
        "src/ngraph_assign_clusters.cc",
        "src/ngraph_builder.h",
        "src/ngraph_builder.cc",
        "src/ngraph_capture_variables.h",
        "src/ngraph_capture_variables.cc",
        "src/ngraph_conversions.h",
        "src/ngraph_cluster_manager.h",
        "src/ngraph_cluster_manager.cc",
        "src/ngraph_deassign_clusters.h",
        "src/ngraph_deassign_clusters.cc",
        "src/ngraph_encapsulate_op.cc",
        "src/ngraph_encapsulate_clusters.h",
        "src/ngraph_encapsulate_clusters.cc",
        "src/ngraph_freshness_tracker.h",
        "src/ngraph_freshness_tracker.cc",
        "src/ngraph_mark_for_clustering.h",
        "src/ngraph_mark_for_clustering.cc",
        "src/ngraph_rewrite_pass.cc",
        "src/ngraph_rewrite_for_tracking.h",
        "src/ngraph_rewrite_for_tracking.cc",
        "src/ngraph_tracked_variable.cc",
        "src/ngraph_utils.h",
        "src/ngraph_utils.cc",
        "src/tf_graphcycles.cc",
        "logging/ngraph_log.h",
        "logging/ngraph_log.cc",
        "logging/tf_graph_writer.h",
        "logging/tf_graph_writer.cc",
    ],
    hdrs = [
        "src/tf_graphcycles.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all_proto_text",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:core_cpu_headers_lib",
        "@ngraph//:ngraph_core",
    ],
    copts = [
        "-I external/ngraph_tf/src",
        "-I external/ngraph_tf/logging",
        "-I external/ngraph/src",
    ],
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

tf_cc_test(
    name = "ngraph_tf_tests",
    size = "small",
    srcs = [
        "test/tf_exec.cpp",
        "test/conversions.cpp",
        "test/padding.cpp",
        "test/graph_rewrites/assign_clusters.cc",
        "test/test_utilities.h",
        "test/test_utilities.cpp",
        "test/test_math_ops.cpp",
        "test/test_nn_ops.cpp",
        "test/opexecuter.h",
        "test/opexecuter.cpp",
        "test/main.cpp",
    ],
    deps = [
        ":ngraph_tf",
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "@org_tensorflow//tensorflow/cc:client_session",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
    extra_copts = [
        "-fexceptions ",
        "-I external/ngraph_tf/src",
        "-I external/ngraph_tf/logging",
        "-I external/ngraph/src",
    ],
)
