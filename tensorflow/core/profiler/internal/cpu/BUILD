load("//tensorflow/core/platform:build_config_root.bzl", "if_static")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "host_tracer_utils",
    srcs = ["host_tracer_utils.cc"],
    hdrs = ["host_tracer_utils.h"],
    copts = tf_profiler_copts(),
    visibility = ["//tensorflow/core/profiler:friends"],
    deps = [
        ":traceme_recorder",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:parse_annotation",
        "//tensorflow/core/profiler/utils:tf_op_utils",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "host_tracer",
    srcs = ["host_tracer.cc"],
    copts = tf_profiler_copts(),
    deps = [
        ":host_tracer_utils",
        ":traceme_recorder",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/core/profiler/lib:profiler_factory",
        "//tensorflow/core/profiler/lib:profiler_interface",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:time_utils",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
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
        "//tensorflow/core/profiler/lib:profiler_interface",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "traceme_recorder",
    hdrs = ["traceme_recorder.h"],
    copts = tf_profiler_copts(),
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "//tensorflow/core:lib",
    ] + if_static([
        ":traceme_recorder_impl",
    ]),
)

cc_library(
    name = "traceme_recorder_impl",
    srcs = [
        "traceme_recorder.cc",
        "traceme_recorder.h",
    ],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/core/profiler:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
    alwayslink = True,
)

tf_cc_test(
    name = "traceme_recorder_test",
    srcs = ["traceme_recorder_test.cc"],
    deps = [
        ":traceme_recorder",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/utils:time_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "annotation_stack",
    hdrs = ["annotation_stack.h"],
    copts = tf_profiler_copts(),
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
    ] + if_static([
        ":annotation_stack_impl",
    ]),
)

cc_library(
    name = "annotation_stack_impl",
    srcs = [
        "annotation_stack.cc",
        "annotation_stack.h",
    ],
    copts = tf_profiler_copts(),
    visibility = ["//tensorflow/core/profiler:__pkg__"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,
)

cc_library(
    name = "python_tracer",
    srcs = ["python_tracer.cc"],
    copts = tf_profiler_copts() + ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/core/profiler/lib:profiler_factory",
        "//tensorflow/core/profiler/lib:profiler_interface",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/python/profiler/internal:python_hooks",
    ],
    alwayslink = True,
)

cc_library(
    name = "metadata_collector",
    srcs = ["metadata_collector.cc"],
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service:xla_debug_info_manager",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/core/profiler/lib:profiler_factory",
        "//tensorflow/core/profiler/lib:profiler_interface",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
    ],
    alwayslink = True,
)
