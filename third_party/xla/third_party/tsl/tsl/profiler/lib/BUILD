load("@xla//xla/tsl:tsl.bzl", "if_not_android", "internal_visibility", "nvtx_headers")
load("@xla//xla/tsl:tsl.default.bzl", "filegroup", "get_compatible_with_portable")
load("@xla//xla/tsl/platform:build_config.bzl", "tsl_cc_test")
load("@xla//xla/tsl/platform:build_config_root.bzl", "if_static")
load("@xla//xla/tsl/platform:rules_cc.bzl", "cc_library")
load(
    "@xla//xla/tsl/platform/default:cuda_build_defs.bzl",
    "if_cuda_is_configured",
)
load(
    "@xla//xla/tsl/profiler/builds:build_config.bzl",
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
        "@xla//xla/tsl/profiler/utils:mobile_srcs_no_runtime",
    ],
    compatible_with = get_compatible_with_portable(),
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
        "@xla//xla/tsl/profiler:internal",
    ]),
    deps = [
        ":profiler_interface",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/status",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "profiler_factory",
    hdrs = ["profiler_factory.h"],
    visibility = internal_visibility([
        "@xla//xla/tsl/profiler:internal",
        "@xla//xla/tsl/profiler:xla_profiler_backends",
        "@xla//xla/python:__pkg__",
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
        "@xla//xla/tsl/profiler:internal",
        "//learning/brain/tfrc/executor/stream_executor:__pkg__",
    ]),
    deps = [
        ":profiler_controller",
        ":profiler_interface",
        "//tsl/profiler/protobuf:profiler_options_proto_cc",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
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
        "//tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:macros",
        "@xla//xla/tsl/platform:test",
    ],
)

cc_library(
    name = "profiler_interface",
    hdrs = ["profiler_interface.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "@xla//xla/tsl:internal",
        "@xla//xla/tsl/profiler:internal",
        "@xla//xla/tsl/profiler:xla_profiler_backends",
    ]),
    deps = [
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "profiler_lock",
    srcs = ["profiler_lock.cc"],
    hdrs = ["profiler_lock.h"],
    copts = tf_profiler_copts(),
    visibility = internal_visibility([
        "@xla//xla/tsl/profiler:internal",
        "@xla//xla/tsl/profiler:xla_internal",
    ]),
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:macros",
        "@xla//xla/tsl/platform:statusor",
        "@xla//xla/tsl/util:env_var",
    ],
)

tsl_cc_test(
    name = "profiler_lock_test",
    srcs = ["profiler_lock_test.cc"],
    deps = [
        ":profiler_lock",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:test",
    ],
)

cc_library(
    name = "profiler_session",
    hdrs = ["profiler_session.h"],
    visibility = internal_visibility([
        "@xla//xla/tsl:internal",
        "//third_party/py/jax:__subpackages__",
    ]),
    deps = [
        "//tsl/platform",
        "//tsl/platform:thread_annotations",
        "//tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:status",
        "@xla//xla/tsl/platform:types",
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
        "@xla//xla/tsl/profiler:internal",
    ]),
    deps = [
        "//tsl/platform:thread_annotations",
        "//tsl/profiler/protobuf:profiler_options_proto_cc",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
        "@xla//xla/tsl/platform:errors",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:xplane_builder",
        "@xla//xla/tsl/profiler/utils:xplane_schema",
        "@xla//xla/tsl/profiler/utils:xplane_utils",
    ] + if_not_android([
        ":profiler_collection",
        ":profiler_factory",
        ":profiler_interface",
        ":profiler_lock",
        "//tsl/platform",
        "//tsl/platform:platform_port",
        "@xla//xla/tsl/platform:status",
        "@xla//xla/tsl/profiler/convert:post_process_single_host_xplane",
        "@xla//xla/tsl/profiler/utils:time_utils",
    ]),
    alwayslink = True,
)

cc_library(
    name = "traceme_encode",
    hdrs = ["traceme_encode.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:macros",
    ],
)

tsl_cc_test(
    name = "traceme_encode_test",
    srcs = ["traceme_encode_test.cc"],
    deps = [
        ":traceme_encode",
        "//tsl/platform",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_benchmark",
    ],
)

tf_profiler_pybind_cc_library_wrapper(
    name = "traceme_for_pybind",
    actual = ":traceme",
    visibility = internal_visibility(["@xla//xla/tsl/profiler:xla_internal"]),
)

cc_library(
    name = "traceme",
    hdrs = ["traceme.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":traceme_encode",
        "//tsl/platform",
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/platform:logging",
        "@xla//xla/tsl/platform:macros",
        "@xla//xla/tsl/platform:types",
        "@xla//xla/tsl/profiler/utils:no_init",
    ] + if_not_android([
        "@xla//xla/tsl/profiler/backends/cpu:traceme_recorder",
        "@xla//xla/tsl/profiler/utils:time_utils",
    ]),
)

cc_library(
    name = "nvtx_utils_impl",
    srcs = if_cuda_is_configured(
        ["nvtx_utils.cc"],
        ["nvtx_utils_stub.cc"],
    ),
    hdrs = ["nvtx_utils.h"],
    visibility = ["//visibility:public"],
    deps = if_cuda_is_configured(nvtx_headers()),
)

cc_library(
    name = "nvtx_utils_external",
    srcs = ["nvtx_utils_stub.cc"],
    hdrs = ["nvtx_utils.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nvtx_utils",
    hdrs = ["nvtx_utils.h"],
    visibility = ["//visibility:public"],
    deps = select({
        # Remove this once we clean up all the CUDA specific deps.
        "@xla//xla/tsl:libtpu_on_gce": [":nvtx_utils_external"],
        # Place holder for internal dependencies
        "//conditions:default": [":nvtx_utils_impl"],
    }),
)

cc_library(
    name = "scoped_annotation",
    hdrs = [
        "scoped_annotation.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":nvtx_utils",
        "//tsl/platform",
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/platform:macros",
        "@xla//xla/tsl/platform:types",
    ] + if_not_android([
        "@xla//xla/tsl/profiler/backends/cpu:annotation_stack",
    ]),
)

tsl_cc_test(
    name = "scoped_annotation_test",
    size = "small",
    srcs = ["scoped_annotation_test.cc"],
    deps = [
        ":scoped_annotation",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@xla//xla/tsl/platform:test",
        "@xla//xla/tsl/platform:test_benchmark",
        "@xla//xla/tsl/profiler/backends/cpu:annotation_stack",
        "@xla//xla/tsl/profiler/backends/cpu:annotation_stack_impl",
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
        "@xla//xla/tsl/platform:types",
    ],
)

cc_library(
    name = "profiler_collection",
    srcs = ["profiler_collection.cc"],
    hdrs = ["profiler_collection.h"],
    visibility = internal_visibility([
        "@xla//xla/backends/profiler/plugin:__pkg__",
        "//learning/brain/tfrc/executor/stream_executor:__pkg__",
        "@xla//xla/backends/profiler/cpu:__pkg__",
        "@xla//xla/backends/profiler/subprocess:__pkg__",
    ]),
    deps = [
        ":profiler_interface",
        "//tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/status",
        "@xla//xla/tsl/platform:status",
    ],
)

cc_library(
    name = "scoped_memory_debug_annotation",
    srcs = ["scoped_memory_debug_annotation.cc"],
    hdrs = ["scoped_memory_debug_annotation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings:string_view",
    ],
)
