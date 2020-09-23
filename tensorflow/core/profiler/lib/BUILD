load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/core/platform:build_config_root.bzl", "if_static")
load("//tensorflow:tensorflow.bzl", "if_not_android", "tf_cc_test", "tf_cuda_library")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")

# buildifier: disable=same-origin-load
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
        "//tensorflow/core/profiler:profiler_options_proto_cc",
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
        "//tensorflow/core/platform",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "@com_google_absl//absl/memory",
        "//tensorflow/core:protos_all_cc",
    ] + if_not_android([
        ":profiler_lock",
        "//tensorflow/core/profiler/convert:post_process_single_host_xplane",
        "//tensorflow/core/profiler/internal:profiler_factory",
        "//tensorflow/core/profiler/utils:derived_timeline",
        "//tensorflow/core/profiler/utils:group_events",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "//tensorflow/core/profiler/utils:xplane_schema",
    ]),
    alwayslink = True,
)

tf_pybind_cc_library_wrapper(
    name = "local_profiler_headers",
    visibility = [
        "//tensorflow/core/profiler/rpc:__pkg__",
        "//tensorflow/python/profiler/internal:__pkg__",
    ],
    deps = [":local_profiler"],
)

cc_library(
    name = "local_profiler",
    hdrs = ["local_profiler.h"],
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "@com_google_absl//absl/memory",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/internal:profiler_interface",
    ] + if_static([
        ":local_profiler_impl",
    ]),
)

cc_library(
    name = "local_profiler_impl",
    srcs = ["local_profiler.cc"],
    hdrs = ["local_profiler.h"],
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        ":profiler_lock",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/core/profiler/convert:post_process_single_host_xplane",
        "//tensorflow/core/profiler/internal:profiler_factory",
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler/protobuf:xplane_proto_cc",
        "//tensorflow/core/profiler/utils:derived_timeline",
        "//tensorflow/core/profiler/utils:group_events",
        "//tensorflow/core/profiler/utils:xplane_schema",
        "//tensorflow/core/profiler/utils:xplane_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/time",
    ],
    alwayslink = True,
)

tf_cuda_library(
    name = "profiler_backends",
    cuda_deps = [
        "//tensorflow/core/profiler/internal/gpu:device_tracer",
        # Metadata collector does not compile for Cast,
        # due to its dependency on human_readable_json.
        # Also, it is only used for XLA:GPU support and
        # thus move it inside cuda_deps.
        # TODO(b/155130190): Gate deps and cuda_deps here so they are empty
        # when compiling for a mobile platform.
        "//tensorflow/core/profiler/internal/cpu:metadata_collector",
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
        ":traceme_encode",
        "@com_google_absl//absl/strings",
        "//tensorflow/core:lib",
        "//tensorflow/core/platform",
    ] + if_not_android([
        "//tensorflow/core/profiler/internal:traceme_recorder",
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
        ":traceme",
        ":traceme_encode",
        "//tensorflow/core:lib",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
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
        "//tensorflow/core/profiler/internal:annotation_stack",
    ]),
)

cc_library(
    name = "profiler_lock",
    srcs = ["profiler_lock.cc"],
    hdrs = ["profiler_lock.h"],
    visibility = ["//tensorflow/core/profiler:internal"],
)

filegroup(
    name = "mobile_srcs",
    srcs = [
        "annotated_traceme.h",
        "connected_traceme.h",
        "profiler_session.cc",
        "profiler_session.h",
        "scoped_annotation.h",
        "traceme.h",
        "traceme_encode.h",
    ],
    visibility = ["//visibility:public"],
)
