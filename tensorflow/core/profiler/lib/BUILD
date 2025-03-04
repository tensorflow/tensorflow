load("//tensorflow:tensorflow.bzl", "if_cuda_or_rocm", "if_not_android", "tf_cc_test", "tf_cuda_library")
load("//tensorflow:tensorflow.default.bzl", "filegroup")
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")
load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow/core/profiler/builds:build_config.bzl",
    "tf_profiler_copts",
    "tf_profiler_pybind_cc_library_wrapper",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
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
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
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
        "//tensorflow/core:lib",
        "//tensorflow/core/platform",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@local_tsl//tsl/profiler/lib:profiler_session",
    ] + if_not_android([
        ":profiler_interface",
        ":profiler_lock",
    ]) + if_static([
        "@local_tsl//tsl/profiler/lib:profiler_session_impl",
    ]),
)

alias(
    name = "profiler_session_impl",
    actual = "@local_tsl//tsl/profiler/lib:profiler_session_impl",
)

cc_library(
    name = "profiler_factory",
    hdrs = ["profiler_factory.h"],
    visibility = ["//tensorflow/c/experimental/pluggable_profiler:__pkg__"],
    deps = [
        ":profiler_interface",
        "@com_google_absl//absl/base:core_headers",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
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
        ":profiler_interface",
        "@com_google_absl//absl/base:core_headers",
        "@local_tsl//tsl/profiler/lib:profiler_factory",
        "@local_tsl//tsl/profiler/lib:profiler_factory_impl",
        "@local_tsl//tsl/profiler/protobuf:profiler_options_proto_cc",
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
        "@com_google_absl//absl/base:core_headers",
        "@local_tsl//tsl/profiler/lib:profiler_interface",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "profiler_controller",
    hdrs = ["profiler_controller.h"],
    deps = [
        ":profiler_interface",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base:core_headers",
        "@local_tsl//tsl/profiler/lib:profiler_controller",
        "@local_tsl//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

tf_cuda_library(
    name = "profiler_backends",
    visibility = ["//tensorflow:internal"],
    deps = [
        "@local_xla//xla/backends/profiler:profiler_backends",
    ] + if_cuda_or_rocm([
        "@local_xla//xla/backends/profiler/gpu:device_tracer",
    ]),
    alwayslink = True,
)

cc_library(
    name = "traceme",
    hdrs = ["traceme.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":traceme_encode",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/lib:traceme",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
    ] + if_not_android([
        "@local_xla//xla/tsl/profiler/backends/cpu:traceme_recorder",
        "@local_xla//xla/tsl/profiler/utils:time_utils",
    ]),
)

cc_library(
    name = "traceme_encode",
    hdrs = ["traceme_encode.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@local_tsl//tsl/profiler/lib:traceme_encode",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@local_tsl//tsl/profiler/lib:connected_traceme",
    ],
)

cc_library(
    name = "context_types_hdrs",
    hdrs = ["context_types.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@local_tsl//tsl/profiler/lib:context_types_hdrs",
    ],
)

cc_library(
    name = "context_types",
    hdrs = ["context_types.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@local_tsl//tsl/profiler/lib:context_types",
    ],
)

cc_library(
    name = "scoped_memory_debug_annotation",
    hdrs = ["scoped_memory_debug_annotation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@local_tsl//tsl/profiler/lib:scoped_memory_debug_annotation",
    ],
)

# TODO(csigg): Remove this forwarding target.
cc_library(
    name = "scoped_annotation",
    hdrs = ["scoped_annotation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@local_tsl//tsl/profiler/lib:scoped_annotation",
    ] + if_not_android([
        "@local_xla//xla/tsl/profiler/backends/cpu:annotation_stack",
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
        "@com_google_absl//absl/base:core_headers",
        "@local_tsl//tsl/profiler/lib:profiler_lock",
    ],
)

tf_cc_test(
    name = "profiler_disabled_test",
    srcs = ["profiler_disabled_test.cc"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "@com_google_absl//absl/status:statusor",
        "@local_tsl//tsl/profiler/lib:profiler_lock",
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
