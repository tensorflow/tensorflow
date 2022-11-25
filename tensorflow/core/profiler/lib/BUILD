load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")
load("//tensorflow:tensorflow.bzl", "if_not_android", "tf_cc_test", "tf_cuda_library")
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load(
    "//tensorflow/core/profiler/builds:build_config.bzl",
    "tf_profiler_copts",
    "tf_profiler_pybind_cc_library_wrapper",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "device_profiler_session",
    hdrs = ["device_profiler_session.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
    ] + if_not_android([
        ":profiler_session",
        "//tensorflow/core/profiler/convert:xplane_to_step_stats",
    ]),
)

tf_profiler_pybind_cc_library_wrapper(
    name = "profiler_session_for_pybind",
    actual = ":profiler_session",
    visibility = ["//tensorflow/python/profiler/internal:__pkg__"],
)

cc_library(
    name = "profiler_session",
    hdrs = ["profiler_session.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform",
        "//tensorflow/tsl/profiler/lib:profiler_session",
    ] + if_not_android([
        ":profiler_interface",
        ":profiler_lock",
    ]) + if_static([
        "//tensorflow/tsl/profiler/lib:profiler_session_impl",
    ]),
)

alias(
    name = "profiler_session_impl",
    actual = "//tensorflow/tsl/profiler/lib:profiler_session_impl",
)

cc_library(
    name = "profiler_factory",
    hdrs = ["profiler_factory.h"],
    deps = [
        ":profiler_interface",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tensorflow/tsl/profiler/lib:profiler_factory",
    ] + if_static([
        ":profiler_factory_impl",
    ]),
)

cc_library(
    name = "profiler_factory_impl",
    srcs = [
        "profiler_factory.h",
    ],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/core/profiler:__pkg__",
    ],
    deps = [
        ":profiler_controller",
        ":profiler_interface",
        "//tensorflow/core:lib",
        "//tensorflow/tsl/profiler/lib:profiler_factory",
        "//tensorflow/tsl/profiler/lib:profiler_factory_impl",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
    ],
    alwayslink = True,
)

cc_library(
    name = "profiler_interface",
    hdrs = ["profiler_interface.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow:internal",
        "//tensorflow/core/profiler:internal",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/profiler/lib:profiler_interface",
    ],
)

cc_library(
    name = "profiler_controller",
    hdrs = ["profiler_controller.h"],
    deps = [
        ":profiler_interface",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/tsl/profiler/lib:profiler_controller",
    ],
)

tf_cuda_library(
    name = "profiler_backends",
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/compiler/xla/backends/profiler:profiler_backends",
    ],
    alwayslink = True,
)

cc_library(
    name = "traceme",
    hdrs = ["traceme.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":traceme_encode",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform",
        "//tensorflow/tsl/profiler/lib:traceme_encode",
        "//tensorflow/tsl/profiler/lib:traceme",
    ] + if_not_android([
        "//tensorflow/core/profiler/backends/cpu:traceme_recorder",
        "//tensorflow/core/profiler/utils:time_utils",
    ]),
)

cc_library(
    name = "traceme_encode",
    hdrs = ["traceme_encode.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/tsl/profiler/lib:traceme_encode",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "annotated_traceme",
    hdrs = ["annotated_traceme.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":scoped_annotation",
        ":traceme",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "connected_traceme",
    hdrs = ["connected_traceme.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":context_types_hdrs",
        ":traceme",
        ":traceme_encode",
        "//tensorflow/core:lib",
        "//tensorflow/tsl/profiler/lib:connected_traceme",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "context_types_hdrs",
    hdrs = ["context_types.h"],
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/tsl/profiler/lib:context_types_hdrs"],
)

cc_library(
    name = "context_types",
    hdrs = ["context_types.h"],
    visibility = ["//visibility:public"],
    deps = ["//tensorflow/tsl/profiler/lib:context_types"],
)

cc_library(
    name = "scoped_memory_debug_annotation",
    hdrs = ["scoped_memory_debug_annotation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tsl/profiler/lib:scoped_memory_debug_annotation",
    ],
)

cc_library(
    name = "scoped_annotation",
    hdrs = ["scoped_annotation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform",
        "//tensorflow/tsl/profiler/lib:scoped_annotation",
    ] + if_not_android([
        "//tensorflow/core/profiler/backends/cpu:annotation_stack",
    ]),
)

cc_library(
    name = "scoped_annotation_stack",
    hdrs = ["scoped_annotation_stack.h"],
    visibility = ["//tensorflow/compiler/xla:runtime"],
    deps = [
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform",
        "//tensorflow/tsl/profiler/lib:scoped_annotation_stack",
    ] + if_not_android([
        "//tensorflow/core/profiler/backends/cpu:annotation_stack",
    ]),
)

cc_library(
    name = "profiler_lock",
    hdrs = ["profiler_lock.h"],
    copts = tf_profiler_copts(),
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/tsl/profiler/lib:profiler_lock",
    ],
)

tf_cc_test(
    name = "profiler_disabled_test",
    srcs = ["profiler_disabled_test.cc"],
    deps = [
        ":profiler_lock",
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "scoped_annotation.h",
        "scoped_memory_debug_annotation.h",
        "traceme.h",
        "traceme_encode.h",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "mobile_srcs_only_runtime",
    srcs = [
        "annotated_traceme.h",
        "connected_traceme.h",
        "context_types.h",
        "device_profiler_session.h",
        "profiler_interface.h",
    ],
    visibility = ["//visibility:public"],
)
