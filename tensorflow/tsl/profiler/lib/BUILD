load("//tensorflow/tsl/platform:rules_cc.bzl", "cc_library")
load("//tensorflow/tsl/platform:build_config_root.bzl", "if_static")
load("//tensorflow/tsl:tsl.default.bzl", "filegroup")
load("//tensorflow:tensorflow.bzl", "if_not_android", "tf_cc_test")
load(
    "//tensorflow/tsl/profiler/builds:build_config.bzl",
    "tf_profiler_copts",
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

filegroup(
    name = "mobile_srcs_no_runtime",
    srcs = [
        "scoped_annotation.h",
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
    visibility = [
        "//tensorflow/core/profiler/lib:__pkg__",
    ],
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
    visibility = [
        "//tensorflow/compiler/xla/backends/profiler:friends",
        "//tensorflow/core/profiler/lib:__pkg__",
    ],
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
        "//tensorflow/core/profiler/lib:__pkg__",
    ],
    deps = [
        ":profiler_interface",
        "//tensorflow/core/profiler:profiler_options_proto_cc",
        "//tensorflow/tsl/platform:mutex",
        "//tensorflow/tsl/profiler/lib:profiler_controller",
    ],
    alwayslink = True,
)

tf_cc_test(
    name = "profiler_factory_test",
    srcs = ["profiler_factory_test.cc"],
    deps = [
        ":profiler_factory",
        ":profiler_factory_impl",
        ":profiler_interface",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:status",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "//tensorflow/tsl/profiler/protobuf:xplane_proto_cc",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "profiler_interface",
    hdrs = ["profiler_interface.h"],
    copts = tf_profiler_copts(),
    visibility = [
        "//tensorflow/compiler/xla/backends/profiler:friends",
        "//tensorflow/core/profiler:internal",
        "//tensorflow/tsl:internal",
    ],
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
    visibility = ["//tensorflow/core/profiler:internal"],
    deps = [
        "//tensorflow/tsl/platform:errors",
        "//tensorflow/tsl/platform:macros",
        "//tensorflow/tsl/platform:statusor",
        "//tensorflow/tsl/util:env_var",
    ],
)

tf_cc_test(
    name = "profiler_lock_test",
    srcs = ["profiler_lock_test.cc"],
    deps = [
        ":profiler_lock",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
    ],
)

cc_library(
    name = "traceme_encode",
    hdrs = ["traceme_encode.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/tsl/platform:logging",
        "//tensorflow/tsl/platform:macros",
        "@com_google_absl//absl/strings",
    ],
)

tf_cc_test(
    name = "traceme_encode_test",
    srcs = ["traceme_encode_test.cc"],
    deps = [
        ":traceme_encode",
        "//tensorflow/tsl/platform",
        "//tensorflow/tsl/platform:test",
        "//tensorflow/tsl/platform:test_main",
        "@com_google_absl//absl/strings",
    ],
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
    name = "scoped_annotation",
    hdrs = ["scoped_annotation.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_google_absl//absl/strings",
        "//tensorflow/tsl/platform:types",
    ] + if_not_android([
        "//tensorflow/tsl/profiler/backends/cpu:annotation_stack",
    ]),
)

cc_library(
    name = "scoped_annotation_stack",
    hdrs = ["scoped_annotation_stack.h"],
    visibility = [
        "//tensorflow/compiler/xla:runtime",
        "//tensorflow/core/profiler/lib:__pkg__",
    ],
    deps = [
        "@com_google_absl//absl/strings",
    ] + if_not_android([
        "//tensorflow/tsl/profiler/backends/cpu:annotation_stack",
    ]),
)

tf_cc_test(
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
