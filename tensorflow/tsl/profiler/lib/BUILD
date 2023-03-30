load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl/platform:build_config_root.bzl", "if_static")
load("//tensorflow/tsl:tsl.default.bzl", "filegroup")
load("//tensorflow/tsl:tsl.bzl", "if_not_android", "set_external_visibility")
load("//tensorflow/tsl/platform:build_config.bzl", "tsl_cc_test")
load(
    "//tensorflow/tsl/profiler/builds:build_config.bzl",
    "tf_profiler_copts",
    "tf_profiler_pybind_cc_library_wrapper",
)

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

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
        "connected_traceme.h",
        "context_types.h",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "profiler_controller",
    srcs = ["profiler_controller.cc"],
    hdrs = ["profiler_controller.h"],
    visibility = set_external_visibility([
        "//tensorflow/tsl/profiler:internal",
    ]),
    deps = [
        ":profiler_interface",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "profiler_factory",
    hdrs = ["profiler_factory.h"],
    visibility = set_external_visibility([
        "//tensorflow/tsl/profiler:internal",
        "//tensorflow/tsl/profiler:xla_profiler_backends",
    ]),
    deps = [
        ":profiler_interface",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
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
    visibility = set_external_visibility([
        "//tensorflow/tsl/profiler:internal",
    ]),
    deps = [
        ":profiler_interface",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/profiler/lib:profiler_controller",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
    ],
    alwayslink = True,
)

tsl_cc_test(
    name = "profiler_factory_test",
    srcs = ["profiler_factory_test.cc"],
    deps = [
        ":profiler_factory",
        ":profiler_factory_impl",
        ":profiler_interface",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "profiler_interface",
    hdrs = ["profiler_interface.h"],
    copts = tf_profiler_copts(),
    visibility = set_external_visibility([
        "//tensorflow/tsl:internal",
        "//tensorflow/tsl/profiler:internal",
        "//tensorflow/tsl/profiler:xla_profiler_backends",
    ]),
    deps = [
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "profiler_lock",
    srcs = ["profiler_lock.cc"],
    hdrs = ["profiler_lock.h"],
    copts = tf_profiler_copts(),
    visibility = set_external_visibility(["//tensorflow/tsl/profiler:internal"]),
    deps = [
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/util:env_var",
    ],
)

tsl_cc_test(
    name = "profiler_lock_test",
    srcs = ["profiler_lock_test.cc"],
    deps = [
        ":profiler_lock",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "profiler_session",
    hdrs = ["profiler_session.h"],
    visibility = set_external_visibility(["//tensorflow/tsl:internal"]),
    deps = [
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
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
    visibility = set_external_visibility([
        "//tensorflow/python:__pkg__",
        "//tensorflow/tsl/profiler:internal",
    ]),
    deps = [
        "@com_google_absl//absl/memory",
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/platform:thread_annotations",
        "//tensorflow/tsl/platform:types",
        "//tensorflow/tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
    ] + if_not_android([
        ":profiler_collection",
        ":profiler_factory",
        ":profiler_interface",
        ":profiler_lock",
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:platform_port",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/profiler/convert:post_process_single_host_xplane",
        "//tensorflow/tsl/profiler/utils:time_utils",
    ]),
    alwayslink = True,
)

cc_library(
    name = "traceme_encode",
    hdrs = ["traceme_encode.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

tsl_cc_test(
    name = "traceme_encode_test",
    srcs = ["traceme_encode_test.cc"],
    deps = [
        ":traceme_encode",
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_profiler_pybind_cc_library_wrapper(
    name = "traceme_for_pybind",
    actual = ":traceme",
    visibility = set_external_visibility(["//tensorflow/tsl/profiler:xla_internal"]),
)

cc_library(
    name = "traceme",
    hdrs = ["traceme.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":traceme_encode",
        "@com_google_absl//absl/strings",
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:types",
    ] + if_not_android([
        "//tensorflow/tsl/profiler/backends/cpu:traceme_recorder",
        "//tensorflow/tsl/profiler/utils:time_utils",
    ]),
)

cc_library(
    name = "nvtx_utils",
    hdrs = ["nvtx_utils.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:types",
    ] + if_not_android([
        "//tensorflow/tsl/profiler/backends/cpu:annotation_stack",
    ]),
)

cc_library(
    name = "scoped_annotation",
    hdrs = ["scoped_annotation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:types",
    ] + if_not_android([
        ":nvtx_utils",
        "//tensorflow/tsl/profiler/backends/cpu:annotation_stack",
    ]),
)

cc_library(
    name = "scoped_annotation_stack",
    hdrs = ["scoped_annotation_stack.h"],
    visibility = set_external_visibility([
        "//tensorflow/tsl/profiler:internal",
        "//tensorflow/tsl/profiler:xla_internal",
    ]),
    deps = [
        "@com_google_absl//absl/strings",
    ] + if_not_android([
        ":nvtx_utils",
        "//tensorflow/tsl/profiler/backends/cpu:annotation_stack",
    ]),
)

tsl_cc_test(
    name = "scoped_annotation_test",
    size = "small",
    srcs = ["scoped_annotation_test.cc"],
    deps = [
        ":scoped_annotation",
        ":scoped_annotation_stack",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_benchmark",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/profiler/backends/cpu:annotation_stack",
        "//tensorflow/tsl/profiler/backends/cpu:annotation_stack_impl",
        "@com_google_absl//absl/strings",
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
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "profiler_collection",
    srcs = ["profiler_collection.cc"],
    hdrs = ["profiler_collection.h"],
    deps = [
        ":profiler_interface",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "scoped_memory_debug_annotation",
    srcs = ["scoped_memory_debug_annotation.cc"],
    hdrs = ["scoped_memory_debug_annotation.h"],
    visibility = ["//visibility:public"],
    deps = [
    ],
)
