load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")
load("//tensorflow:tensorflow.bzl", "if_libtpu", "if_not_android", "if_xla_available", "tf_cc_test", "tf_cuda_library")
load("//tensorflow:tensorflow.bzl", "filegroup")
load(
    "//tensorflow/core/profiler/builds:build_config.bzl",
    "tf_profiler_copts",
    "tf_profiler_pybind_cc_library_wrapper",
)

package(
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
        "//tensorflow/core/profiler:profiler_options_proto_cc",
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
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform",
    ] + if_not_android([
        ":profiler_interface",
        ":profiler_lock",
    ]) + if_static([
        ":profiler_session_impl",
    ]),
)

cc_library(
    name = "profiler_session_impl",
    srcs = [
        "profiler_session.cc",
        "profiler_session.h",
    ],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/core/profiler:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/platform",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "@com_google_absl//absl/memory",
    ] + if_not_android([
        "//tensorflow/core:lib_internal",
        ":profiler_interface",
        ":profiler_factory",
        ":profiler_collection",
        ":profiler_lock",
        "//tensorflow/core/profiler/convert:post_process_single_host_xplane",
        "//tensorflow/core/profiler/utils:time_utils",
    ]),
    alwayslink = True,
)

cc_library(
    name = "profiler_factory",
    hdrs = ["profiler_factory.h"],
    deps = [
        ":profiler_interface",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
    ] + if_static([
        ":profiler_factory_impl",
    ]),
)

cc_library(
    name = "profiler_factory_impl",
    srcs = [
        "profiler_factory.cc",
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
        "//tensorflow/core/profiler:profiler_options_proto_cc",
    ],
    alwayslink = True,
)

tf_cc_test(
    name = "profiler_factory_test",
    srcs = ["profiler_factory_test.cc"],
    deps = [
        ":profiler_factory",
        ":profiler_interface",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/memory",
    ],
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
    ],
)

cc_library(
    name = "profiler_collection",
    srcs = ["profiler_collection.cc"],
    hdrs = ["profiler_collection.h"],
    deps = [
        ":profiler_interface",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "profiler_controller",
    srcs = ["profiler_controller.cc"],
    hdrs = ["profiler_controller.h"],
    deps = [
        ":profiler_interface",
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
    ],
)

tf_cuda_library(
    name = "profiler_backends",
    cuda_deps = [
        "//tensorflow/core/profiler/backends/gpu:device_tracer",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/profiler/backends/cpu:host_tracer",
    ] + if_libtpu([
        "//tensorflow/core/profiler/backends/tpu:tpu_tracer",
    ]) + if_xla_available([
        "//tensorflow/core/profiler/backends/cpu:metadata_collector",
    ]),
    alwayslink = True,
)

tf_profiler_pybind_cc_library_wrapper(
    name = "traceme_for_pybind",
    actual = ":traceme",
    visibility = ["//tensorflow/python/profiler/internal:__pkg__"],
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
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "traceme_encode_test",
    srcs = ["traceme_encode_test.cc"],
    deps = [
        ":traceme_encode",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/platform",
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
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "context_types_hdrs",
    hdrs = ["context_types.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "context_types",
    srcs = ["context_types.cc"],
    hdrs = ["context_types.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "scoped_memory_debug_annotation",
    srcs = ["scoped_memory_debug_annotation.cc"],
    hdrs = ["scoped_memory_debug_annotation.h"],
    visibility = ["//visibility:public"],
    deps = [
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
    ] + if_not_android([
        "//tensorflow/core/profiler/backends/cpu:annotation_stack",
    ]),
)

tf_cc_test(
    name = "scoped_annotation_test",
    size = "small",
    srcs = ["scoped_annotation_test.cc"],
    deps = [
        ":scoped_annotation",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core/profiler/backends/cpu:annotation_stack",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "profiler_lock",
    srcs = ["profiler_lock.cc"],
    hdrs = ["profiler_lock.h"],
    copts = tf_profiler_copts(),
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

tf_cc_test(
    name = "profiler_lock_test",
    srcs = ["profiler_lock_test.cc"],
    deps = [
        ":profiler_lock",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
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
        "scoped_memory_debug_annotation.cc",
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
