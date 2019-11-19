load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/platform:default/build_config_root.bzl", "if_static")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "host_tracer",
    srcs = ["host_tracer.cc"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/internal:parse_annotation",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/internal:traceme_recorder",
        "//tensorflow/core/profiler/utils:xplane_builder",
    ] + if_static(
        extra_deps = ["//tensorflow/core/profiler/protobuf:xplane_proto_cc"],
        otherwise = ["//tensorflow/core/profiler/protobuf:xplane_proto_cc_headers_only"],
    ),
    alwayslink = True,
)

tf_cc_test(
    name = "host_tracer_test",
    srcs = ["host_tracer_test.cc"],
    deps = [
        ":host_tracer",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest_main",
    ],
)
