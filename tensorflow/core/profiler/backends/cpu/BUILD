load("//tensorflow/core/platform:build_config_root.bzl", "if_static")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow:tensorflow.bzl", "tf_cc_test")
load("//tensorflow/core/profiler/builds:build_config.bzl", "tf_profiler_copts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "host_tracer_utils",
    hdrs = ["host_tracer_utils.h"],
    copts = tf_profiler_copts(),
    deps = [
        ":traceme_recorder",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/profiler/backends/cpu:host_tracer_utils",
    ],
)

cc_library(
    name = "host_tracer",
    deps = [
        "//tensorflow/compiler/xla/backends/profiler/cpu:host_tracer",
    ],
    alwayslink = True,
)

cc_library(
    name = "host_tracer_impl",
    hdrs = ["host_tracer.h"],
    copts = tf_profiler_copts(),
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "//tensorflow/compiler/xla/backends/profiler/cpu:host_tracer_impl",
        "//tensorflow/core/profiler/lib:profiler_interface",
    ],
)

tf_cc_test(
    name = "host_tracer_test",
    srcs = ["host_tracer_test.cc"],
    deps = [
        ":host_tracer_impl",
        "//tensorflow/core:core_cpu_lib",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/lib:profiler_interface",
        "//tensorflow/core/profiler/lib:profiler_session",
        "//tensorflow/core/profiler/lib:traceme",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_visitor",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
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
        "//tensorflow/core:lib",
        "//tensorflow/tsl/profiler/backends/cpu:traceme_recorder",
        "@com_google_absl//absl/container:flat_hash_map",
    ] + if_static([
        ":traceme_recorder_impl",
    ]),
)

cc_library(
    name = "traceme_recorder_impl",
    hdrs = [
        "traceme_recorder.h",
    ],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/core/profiler:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/tsl/profiler/backends/cpu:traceme_recorder_impl",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
    alwayslink = True,
)

cc_library(
    name = "annotation_stack",
    hdrs = ["annotation_stack.h"],
    copts = tf_profiler_copts(),
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/tsl/profiler/backends/cpu:annotation_stack",
        "@com_google_absl//absl/strings",
    ] + if_static([
        ":annotation_stack_impl",
    ]),
)

cc_library(
    name = "annotation_stack_impl",
    srcs = [
        "annotation_stack.h",
    ],
    copts = tf_profiler_copts(),
    visibility = ["//tensorflow/core/profiler:__pkg__"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/tsl/profiler/backends/cpu:annotation_stack",
        "//tensorflow/tsl/profiler/backends/cpu:annotation_stack_impl",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,
)

cc_library(
    name = "python_tracer",
    deps = [
        "//tensorflow/compiler/xla/backends/profiler/cpu:python_tracer",
    ],
    alwayslink = True,
)

cc_library(
    name = "python_tracer_impl",
    hdrs = ["python_tracer.h"],
    copts = tf_profiler_copts() + ["-fexceptions"],
    features = ["-use_header_modules"],
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "//tensorflow/compiler/xla/backends/profiler/cpu:python_tracer_impl",
        "//tensorflow/tsl/profiler/lib:profiler_interface",
    ],
)

cc_library(
    name = "metadata_collector",
    copts = tf_profiler_copts(),
    deps = [
        "//tensorflow/compiler/xla/backends/profiler/cpu:metadata_collector",
    ],
    alwayslink = True,
)

cc_library(
    name = "metadata_utils",
    hdrs = ["metadata_utils.h"],
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "//tensorflow/compiler/xla/backends/profiler/cpu:metadata_utils",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/core/profiler/convert:xla_op_utils",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:xplane_builder",
        "//tensorflow/core/profiler/utils:xplane_schema",
    ],
)
