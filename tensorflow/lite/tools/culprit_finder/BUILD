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
    name = "model_metadata_lib",
    srcs = ["model_metadata_lib.cc"],
    hdrs = ["model_metadata_lib.h"],
    copts = ["-fexceptions"],
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],  # Incompatible with -fexceptions.
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/profiling:model_runtime_info",
        "//tensorflow/lite/tools:logging",
        "//third_party/odml/litert/litert/c:litert_common",
        "//third_party/odml/litert/litert/cc:litert_expected",
    ],
)

cc_library(
    name = "tflite_input_manager",
    srcs = ["tflite_input_manager.cc"],
    hdrs = ["tflite_input_manager.h"],
    copts = ["-fexceptions"],
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],  # Incompatible with -fexceptions.
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/tools:utils",
    ],
)

cc_library(
    name = "interpreter_handler",
    srcs = ["interpreter_handler.cc"],
    hdrs = ["interpreter_handler.h"],
    copts = ["-fexceptions"],
    features = [
        "-use_header_modules",
        "-parse_headers",
    ],  # Incompatible with -fexceptions.
    visibility = ["//visibility:public"],
    deps = [
        ":tflite_input_manager",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:framework_stable",
        "//tensorflow/lite/c:c_api_types",
        "//tensorflow/lite/core:model_builder",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/tools:model_loader",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
        "//third_party/odml/litert/litert/c:litert_common",
        "//third_party/odml/litert/litert/cc:litert_expected",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
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
        ":interpreter_handler",
        ":model_metadata_lib",
        ":tflite_input_manager",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:span",
        # copybara:uncomment_begin(google-only)
        # "//third_party/odml/infra/ml_drift_delegate:ml_drift_delegate_provider",
        # "//third_party/odml/litert/litert/cc:litert_expected",
        # "//third_party/odml/litert/litert/cc:litert_macros",
        # "//third_party/odml/litert/litert/cc:litert_tflite_error_status_builder",
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
