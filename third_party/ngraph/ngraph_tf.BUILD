licenses(["notice"])  # 3-Clause BSD

exports_files(["LICENSE"])

load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

cc_library(
    name = "ngraph_libs_linux",
    srcs = [
        "lib/libiomp5.so",
        "lib/libmklml_intel.so",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ngraph_tf",
    srcs = [
        "src/ngraph_builder.h",
        "src/ngraph_builder.cc",
        "src/ngraph_cluster.h",
        "src/ngraph_cluster.cc",
        "src/ngraph_cluster_manager.h",
        "src/ngraph_cluster_manager.cc",
        "src/ngraph_confirm_pass.cc",
        "src/ngraph_device.cc",
        "src/ngraph_encapsulate_op.cc",
        "src/ngraph_encapsulate_pass.cc",
        "src/ngraph_freshness_tracker.h",
        "src/ngraph_freshness_tracker.cc",
        "src/ngraph_graph_rewrite_passes.cc",
        "src/ngraph_liberate_pass.cc",
        "src/ngraph_op_kernels.cc",
        "src/ngraph_stub_ops.cc",
        "src/ngraph_utils.h",
        "src/ngraph_utils.cc",
        "src/ngraph_send_recv_ops.cc",
        "src/ngraph_variable_ops.cc",
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
        "-D NGRAPH_EMBEDDED_IN_TENSORFLOW=1",
    ],
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

tf_cc_test(
    name = "ngraph_tf_tests",
    size = "small",
    srcs = [
        "test/tf_exec.cpp",
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
        "-D NGRAPH_EMBEDDED_IN_TENSORFLOW=1",
        "-I external/ngraph_tf/src",
        "-I external/ngraph_tf/logging",
        "-I external/ngraph/src",
    ],
)
