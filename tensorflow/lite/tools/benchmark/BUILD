load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

exports_files(["logging.h"])

common_copts = ["-Wall"] + tflite_copts()

cc_library(
    name = "logging",
    hdrs = ["logging.h"],
    copts = common_copts,
)

cc_binary(
    name = "benchmark_model",
    srcs = [
        "benchmark_main.cc",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-Wl,--rpath=/data/local/tmp/",  # Hexagon delegate libraries should be in /data/local/tmp
        ],
        "//conditions:default": [],
    }),
    tags = ["builder_default_android_arm64"],
    deps = [
        ":benchmark_tflite_model_lib",
        ":logging",
    ],
)

cc_binary(
    name = "benchmark_model_performance_options",
    srcs = [
        "benchmark_tflite_performance_options_main.cc",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    tags = ["builder_default_android_arm64"],
    deps = [
        ":benchmark_performance_options",
        ":benchmark_tflite_model_lib",
        ":logging",
    ],
)

# As with most target binaries that use flex, this should be built with the
# `--config=monolithic` build flag, e.g.,
#    bazel build --config=monolithic --config=android_arm64 \
#        -c opt --cxxopt='--std=c++14' :benchmark_model_plus_flex
tf_cc_binary(
    name = "benchmark_model_plus_flex",
    srcs = [
        "benchmark_plus_flex_main.cc",
    ],
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":benchmark_tflite_model_lib",
        ":logging",
        "//tensorflow/lite/delegates/flex:delegate",
        "//tensorflow/lite/testing:init_tensorflow",
    ],
)

cc_test(
    name = "benchmark_test",
    srcs = ["benchmark_test.cc"],
    args = [
        "--fp32_graph=$(location //tensorflow/lite:testdata/multi_add.bin)",
        "--int8_graph=$(location //tensorflow/lite:testdata/add_quantized_int8.bin)",
        "--string_graph=$(location //tensorflow/lite:testdata/string_input_model.bin)",
    ],
    data = [
        "//tensorflow/lite:testdata/add_quantized_int8.bin",
        "//tensorflow/lite:testdata/multi_add.bin",
        "//tensorflow/lite:testdata/string_input_model.bin",
    ],
    tags = [
        "tflite_not_portable_android",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":benchmark_performance_options",
        ":benchmark_tflite_model_lib",
        ":delegate_provider_hdr",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/testing:util",
        "//tensorflow/lite/tools:command_line_flags",
        "@com_google_absl//absl/algorithm",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest",
    ],
)

cc_library(
    name = "profiling_listener",
    srcs = ["profiling_listener.cc"],
    hdrs = ["profiling_listener.h"],
    copts = common_copts,
    deps = [
        ":benchmark_model_lib",
        "//tensorflow/lite/profiling:profile_summarizer",
        "//tensorflow/lite/profiling:profile_summary_formatter",
        "//tensorflow/lite/profiling:profiler",
    ],
)

cc_library(
    name = "benchmark_tflite_model_lib",
    srcs = ["benchmark_tflite_model.cc"],
    hdrs = ["benchmark_tflite_model.h"],
    copts = common_copts + select({
        "//tensorflow:ios": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":profiling_listener",
        ":benchmark_model_lib",
        ":benchmark_utils",
        ":delegate_provider_hdr",
        ":gpu_delegate_provider",
        ":hexagon_delegate_provider",
        ":external_delegate_provider",
        ":logging",
        ":nnapi_delegate_provider",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite/experimental/ruy/profiler",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/profiling:platform_profiler",
        "//tensorflow/lite/profiling:profiler",
        "//tensorflow/lite/profiling:profile_summary_formatter",
        "//tensorflow/lite/tools/evaluation:utils",
    ] + select({
        "//tensorflow:fuchsia": [],
        "//conditions:default": [
            ":xnnpack_delegate_provider",
        ],
    }),
)

cc_library(
    name = "benchmark_performance_options",
    srcs = [
        "benchmark_performance_options.cc",
    ],
    hdrs = ["benchmark_performance_options.h"],
    copts = common_copts,
    deps = [
        ":benchmark_model_lib",
        ":benchmark_params",
        ":benchmark_utils",
        ":logging",
        "//tensorflow/core/util:stats_calculator_portable",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/nnapi:nnapi_util",
        "//tensorflow/lite/profiling:time",
        "//tensorflow/lite/tools:command_line_flags",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/lite/delegates/gpu:delegate",
        ],
        "//conditions:default": [],
    }),
)

cc_library(
    name = "benchmark_params",
    srcs = [
        "benchmark_params.cc",
    ],
    hdrs = ["benchmark_params.h"],
    copts = common_copts,
    deps = [":logging"],
)

cc_library(
    name = "benchmark_model_lib",
    srcs = [
        "benchmark_model.cc",
    ],
    hdrs = ["benchmark_model.h"],
    copts = common_copts,
    deps = [
        ":benchmark_params",
        ":benchmark_utils",
        ":logging",
        "//tensorflow/core/util:stats_calculator_portable",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/profiling:memory_info",
        "//tensorflow/lite/profiling:time",
        "//tensorflow/lite/tools:command_line_flags",
    ],
)

cc_library(
    name = "delegate_provider_hdr",
    hdrs = [
        "delegate_provider.h",
    ],
    copts = common_copts,
    deps = [
        ":benchmark_params",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
    ],
)

cc_library(
    name = "gpu_delegate_provider",
    srcs = ["gpu_delegate_provider.cc"],
    copts = common_copts + select({
        "//tensorflow:ios": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":benchmark_model_lib",
        ":benchmark_params",
        ":delegate_provider_hdr",
        ":logging",
        "//tensorflow/lite/tools/evaluation:utils",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/lite/delegates/gpu:delegate",
        ],
        "//tensorflow:ios": [
            "//tensorflow/lite/delegates/gpu:metal_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,
)

cc_library(
    name = "nnapi_delegate_provider",
    srcs = ["nnapi_delegate_provider.cc"],
    copts = common_copts,
    deps = [
        ":benchmark_model_lib",
        ":benchmark_params",
        ":delegate_provider_hdr",
        ":logging",
        "//tensorflow/lite/tools/evaluation:utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "hexagon_delegate_provider",
    srcs = ["hexagon_delegate_provider.cc"],
    copts = common_copts,
    deps = [
        ":benchmark_model_lib",
        ":benchmark_params",
        ":delegate_provider_hdr",
        ":logging",
        "//tensorflow/lite/tools/evaluation:utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "xnnpack_delegate_provider",
    srcs = ["xnnpack_delegate_provider.cc"],
    copts = tflite_copts(),
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        ":benchmark_model_lib",
        ":delegate_provider_hdr",
        ":logging",
        "//tensorflow/lite/tools/evaluation:utils",
    ],
    alwayslink = 1,
)

cc_library(
    name = "external_delegate_provider",
    srcs = ["external_delegate_provider.cc"],
    copts = tflite_copts(),
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        ":benchmark_model_lib",
        ":delegate_provider_hdr",
        ":logging",
    ],
    alwayslink = 1,
)

cc_library(
    name = "benchmark_utils",
    srcs = [
        "benchmark_utils.cc",
    ],
    hdrs = ["benchmark_utils.h"],
    copts = common_copts,
    deps = ["//tensorflow/lite/profiling:time"],
)

cc_test(
    name = "benchmark_utils_test",
    srcs = [
        "benchmark_utils_test.cc",
    ],
    copts = common_copts,
    deps = [
        ":benchmark_utils",
        "//tensorflow/lite/profiling:time",
        "@com_google_googletest//:gtest_main",
    ],
)

tflite_portable_test_suite()
