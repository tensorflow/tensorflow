package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "culprit_finder_utils",
    hdrs = ["culprit_finder_utils.h"],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:cc_api",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/tools:tool_params",
    ],
)

cc_library(
    name = "culprit_finder_lib",
    srcs = ["culprit_finder_lib.cc"],
    hdrs = ["culprit_finder_lib.h"],
    copts = ["-fexceptions"],
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],  # Incompatible with -fexceptions.
    visibility = ["//visibility:public"],
    deps = [
        ":culprit_finder_utils",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        # copybara:uncomment_begin(google-only)
        # "//third_party/odml/infra/ml_drift_delegate:ml_drift_delegate_provider",
        # "//tensorflow/lite/profiling/proto:model_runtime_info_cc",
        # copybara:uncomment_end
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core:cc_api",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/profiling:memory_usage_monitor",
        "//tensorflow/lite/profiling:model_runtime_info",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/tools:model_loader",
        "//tensorflow/lite/tools:tool_params",
        "//tensorflow/lite/tools:utils",
        "//tensorflow/lite/tools/delegates:default_execution_provider",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
    ] + select({
        # Metal GPU delegate for iOS has its own setups for kernel tests, so
        # skipping linking w/ the gpu_delegate_provider.
        "//tensorflow:ios": [],
        "//conditions:default": [
            "//tensorflow/lite/tools/delegates:gpu_delegate_provider",
        ],
    }),
)

cc_binary(
    name = "culprit_finder_main",
    srcs = ["culprit_finder_main.cc"],
    deps = [
        ":culprit_finder_lib",
        "//tensorflow/lite/tools:logging",
    ],
)
