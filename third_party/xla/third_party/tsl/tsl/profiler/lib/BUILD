load("@local_tsl//tsl/platform:rules_cc.bzl", "cc_library")
load("@local_xla//xla/tsl:tsl.bzl", "if_not_android", "internal_visibility", "nvtx_headers")
load("@local_xla//xla/tsl:tsl.default.bzl", "filegroup")
load("//tsl/platform:build_config.bzl", "tsl_cc_test")
load("//tsl/platform:build_config_root.bzl", "if_static")
load(
    "//tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "//tsl/profiler/builds:build_config.bzl",
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
        "nvtx_utils.h",
        "nvtx_utils_stub.cc",  #Include the stub implementation here since CUDA isn't relevant to Android.
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
    visibility = internal_visibility([
        "//tsl/profiler:internal",
    ]),
    deps = [
        ":profiler_interface",
        "//tsl/platform:errors",
        "//tsl/platform:logging",
        "//tsl/platform:status",
        "//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "profiler_factory",
    hdrs = ["profiler_factory.h"],
    visibility = internal_visibility([
        "//tsl/profiler:internal",
        "//tsl/profiler:xla_profiler_backends",
        "@local_xla//xla/python:__pkg__",
        "//learning/brain/tfrc/executor/stream_executor:__pkg__",
    ]),
    deps = [
        ":profiler_interface",
        "//tsl/profiler/protobuf:profiler_options_proto_cc",
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
    visibility = internal_visibility([
        "//tsl/profiler:internal",
        "//learning/brain/tfrc/executor/stream_executor:__pkg__",
    ]),
    deps = [
        ":profiler_interface",
        "//tsl/platform:mutex",
        "//tsl/profiler/lib:profiler_controller",
        "//tsl/profiler/protobuf:profiler_options_proto_cc",
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
        "//tsl/platform:macros",
        "//tsl/platform:status",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
        "//tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "profiler_interface",
    hdrs = ["profiler_interface.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "@local_xla//xla/tsl:internal",
        "//tsl/profiler:internal",
        "//tsl/profiler:xla_profiler_backends",
    ]),
    deps = [
        "//tsl/platform:status",
        "//tsl/profiler/protobuf:xplane_proto_cc",
    ],
)

cc_library(
    name = "profiler_lock",
    srcs = ["profiler_lock.cc"],
    hdrs = ["profiler_lock.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "//tsl/profiler:internal",
        "//tsl/profiler:xla_internal",
    ]),
    deps = [
        "//tsl/platform:errors",
        "//tsl/platform:macros",
        "//tsl/platform:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@local_xla//xla/tsl/util:env_var",
    ],
)

tsl_cc_test(
    name = "profiler_lock_test",
    srcs = ["profiler_lock_test.cc"],
    deps = [
        ":profiler_lock",
        "//tsl/platform:statusor",
        "//tsl/platform:test",
        "//tsl/platform:test_main",
    ],
)

cc_library(
    name = "profiler_session",
    hdrs = ["profiler_session.h"],
    visibility = internal_visibility(["@local_xla//xla/tsl:internal"]),
    deps = [
        "//tsl/platform",
        "//tsl/platform:errors",
        "//tsl/platform:mutex",
        "//tsl/platform:status",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tsl/profiler/protobuf:xplane_proto_cc",
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
    visibility = internal_visibility([
        "//tensorflow/python:__pkg__",
        "//tsl/profiler:internal",
    ]),
    deps = [
        "//tsl/platform:errors",
        "//tsl/platform:logging",
        "//tsl/platform:mutex",
        "//tsl/platform:thread_annotations",
        "//tsl/platform:types",
        "//tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/memory",
    ] + if_not_android([
        ":profiler_collection",
        ":profiler_factory",
        ":profiler_interface",
        ":profiler_lock",
        "//tsl/platform",
        "//tsl/platform:platform_port",
        "//tsl/platform:status",
        "//tsl/profiler/convert:post_process_single_host_xplane",
        "//tsl/profiler/utils:time_utils",
    ]),
    alwayslink = True,
)

cc_library(
    name = "traceme_encode",
    hdrs = ["traceme_encode.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

tsl_cc_test(
    name = "traceme_encode_test",
    srcs = ["traceme_encode_test.cc"],
    deps = [
        ":traceme_encode",
        "//tsl/platform",
        "//tsl/platform:test",
        "//tsl/platform:test_benchmark",
        "//tsl/platform:test_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tf_profiler_pybind_cc_library_wrapper(
    name = "traceme_for_pybind",
    actual = ":traceme",
    visibility = internal_visibility(["//tsl/profiler:xla_internal"]),
)

cc_library(
    name = "traceme",
    hdrs = ["traceme.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":traceme_encode",
        "//tsl/platform",
        "//tsl/platform:logging",
        "//tsl/platform:macros",
        "//tsl/platform:types",
        "@com_google_absl//absl/strings",
    ] + if_not_android([
        "//tsl/profiler/backends/cpu:traceme_recorder",
        "//tsl/profiler/utils:time_utils",
    ]),
)

cc_library(
    name = "nvtx_utils",
    srcs = if_cuda_is_configured(
        ["nvtx_utils.cc"],
        ["nvtx_utils_stub.cc"],
    ),
    hdrs = ["nvtx_utils.h"],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured(nvtx_headers()),
)

cc_library(
    name = "scoped_annotation",
    hdrs = [
        "nvtx_utils.h",
        "scoped_annotation.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":nvtx_utils",
        "//tsl/platform:macros",
        "//tsl/platform:types",
        "@com_google_absl//absl/strings",
    ] + if_not_android([
        "//tsl/profiler/backends/cpu:annotation_stack",
    ]),
)

tsl_cc_test(
    name = "scoped_annotation_test",
    size = "small",
    srcs = ["scoped_annotation_test.cc"],
    deps = [
        ":scoped_annotation",
        "//tsl/platform:test",
        "//tsl/platform:test_benchmark",
        "//tsl/platform:test_main",
        "//tsl/profiler/backends/cpu:annotation_stack",
        "//tsl/profiler/backends/cpu:annotation_stack_impl",
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
    visibility = internal_visibility([
        "@local_xla//xla/backends/profiler/plugin:__pkg__",
        "//learning/brain/tfrc/executor/stream_executor:__pkg__",
        "@local_xla//xla/backends/profiler/cpu:__pkg__",
    ]),
    deps = [
        ":profiler_interface",
        "//tsl/platform:status",
        "//tsl/profiler/protobuf:xplane_proto_cc",
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
