load(
    "//tensorflow:tensorflow.bzl",
    "tf_cuda_library",
)
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_additional_profiler_lib_deps",
)

package(
    default_visibility = [
        "//tensorflow:internal",
        "//tensorflow_models:__subpackages__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

tf_cuda_library(
    name = "profiler_session",
    srcs = [
        "profiler_session.cc",
    ],
    hdrs = [
        "profiler_session.h",
    ],
    visibility = ["//tensorflow:internal"],
    deps = [
        "//tensorflow/core/profiler/internal:profiler_interface",
        "//tensorflow/core/profiler:protos_all_cc",
    ] + select({
        "//tensorflow:android": [],
        "//conditions:default": [
            "//tensorflow/core:core_cpu_lib",
            "//tensorflow/core:framework",
            "//tensorflow/core:framework_internal",
            "//tensorflow/core:lib",
            "//tensorflow/core:lib_internal",
            "//tensorflow/core:protos_all_cc",
            "//tensorflow/core:session_options",
        ],
    }),
)

tf_cuda_library(
    name = "profiler_lib",
    visibility = ["//tensorflow:internal"],
    deps = tf_additional_profiler_lib_deps(),
    alwayslink = 1,
)

tf_cuda_library(
    name = "traceme",
    srcs = ["traceme.cc"],
    hdrs = ["traceme.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core/profiler/internal:traceme_recorder",
        "@com_google_absl//absl/strings",
    ],
)

filegroup(
    name = "mobile_srcs",
    srcs = glob(["*"]),
    visibility = ["//visibility:public"],
)
