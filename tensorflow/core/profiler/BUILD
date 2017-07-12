package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
)

# -----------------------------------------------------------------------------
# Google-internal targets.  These must be at the end for syncrepo.

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

tf_cc_binary(
    name = "profiler",
    srcs = ["profiler.cc"],
    deps = [
        ":protos_all_cc",
        "//tensorflow/c:c_api",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/platform/default/build_config:platformlib",
        "//tensorflow/core/profiler/internal:tfprof_options",
        "//tensorflow/core/profiler/internal:tfprof_stats",
        "//tensorflow/core/profiler/internal:tfprof_utils",
        "//tensorflow/core/profiler/internal/advisor:tfprof_advisor",
        "@linenoise//:linenoise",
        "@protobuf//:protobuf",
    ],
)

load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")

tf_proto_library(
    name = "protos_all",
    srcs = glob(
        ["**/*.proto"],
    ),
    cc_api_version = 2,
    go_api_version = 2,
    java_api_version = 2,
    protodeps = ["//tensorflow/core:protos_all"],
    visibility = ["//visibility:public"],
)
