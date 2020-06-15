load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "host_tracer_utils",
    srcs = ["host_tracer_utils.cc"],
    hdrs = ["host_tracer_utils.h"],
    visibility = ["//tensorflow/core/profiler:friends"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/internal:parse_annotation",
        "//tensorflow/core/profiler/internal:traceme_recorder",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:tf_op_utils",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "host_tracer",
    srcs = ["host_tracer.cc"],
    deps = [
        ":host_tracer_utils",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/core/profiler/internal:profiler_factory",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/internal:traceme_recorder",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "@com_google_absl//absl/strings",
    ],
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
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "python_tracer",
    srcs = ["python_tracer.cc"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/core/profiler/internal:profiler_factory",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/python/profiler/internal:python_hooks",
    ],
    alwayslink = True,
)

cc_library(
    name = "metadata_collector",
    srcs = ["metadata_collector.cc"],
    deps = [
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service/gpu:gpu_debug_info_manager",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/core/profiler/internal:profiler_factory",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
    ],
    alwayslink = True,
)
