load("//tensorflow/core/platform:build_config_root.bzl", "if_static")
load("//tensorflow:tensorflow.bzl", "if_not_android", "tf_cuda_library")
load("//tensorflow:tensorflow.bzl", "tf_pybind_cc_library_wrapper")

package(
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

tf_pybind_cc_library_wrapper(
    name = "profiler_session_headers",
    visibility = [
        "//tensorflow/core/profiler/rpc:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ],
    deps = [":profiler_session"],
)

cc_library(
    name = "profiler_session",
    hdrs = ["profiler_session.h"],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
    ] + if_static([
        ":profiler_session_impl",
    ]),
)

cc_library(
    name = "profiler_session_impl",
    srcs = [
        "profiler_session.cc",
        "profiler_session.h",
    ],
    visibility = [
        "//tensorflow/core/profiler:__pkg__",
        "//tensorflow/python:__pkg__",
    ],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/util:ptr_util",
    ] + if_not_android([
        ":profiler_utils",
        "//tensorflow/core/profiler/internal:profiler_factory",
        "//tensorflow/core/profiler/convert:xplane_to_trace_events",
        "//tensorflow/core/profiler/utils:derived_timeline",
        "//tensorflow/core/profiler/utils:group_events",
        "//tensorflow/core/profiler/utils:xplane_utils",
    ]),
    alwayslink = True,
)

tf_cuda_library(
    name = "profiler_backends",
    cuda_deps = [
        "//tensorflow/core/profiler/internal/gpu:device_tracer",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/profiler/internal/cpu:host_tracer",
    ],
    alwayslink = True,
)

tf_pybind_cc_library_wrapper(
    name = "traceme_headers",
    visibility = ["//tensorflow/python/profiler/internal:__pkg__"],
    deps = [":traceme"],
)

cc_library(
    name = "traceme",
    hdrs = ["traceme.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform",
    ] + if_not_android([
        "//tensorflow/core/profiler/internal:traceme_recorder",
    ]),
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
    ],
)

tf_pybind_cc_library_wrapper(
    name = "scoped_annotation_headers",
    visibility = ["//tensorflow/python/profiler/internal:__pkg__"],
    deps = [":scoped_annotation"],
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
        "//tensorflow/core/profiler/internal:annotation_stack",
    ]),
)

cc_library(
    name = "profiler_utils",
    srcs = ["profiler_utils.cc"],
    hdrs = ["profiler_utils.h"],
    visibility = ["//tensorflow/core/profiler:internal"],
)

filegroup(
    name = "mobile_srcs",
    srcs = [
        "annotated_traceme.h",
        "profiler_session.cc",
        "profiler_session.h",
        "scoped_annotation.h",
        "traceme.h",
    ],
    visibility = ["//visibility:public"],
)
