package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

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

cc_binary(
    name = "tfprof",
    srcs = ["tfprof_main.cc"],
    deps = [
        ":protos_all_cc",
        "//tensorflow/c:c_api",
        "//tensorflow/c:checkpoint_reader",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core/profiler/internal:tfprof_options",
        "//tensorflow/core/profiler/internal:tfprof_stats",
        "//tensorflow/core/profiler/internal:tfprof_utils",
        "//tensorflow/core/profiler/internal/advisor:tfprof_advisor",
        "@linenoise//:linenoise",
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
