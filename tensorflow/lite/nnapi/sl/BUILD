load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")
load("//tensorflow/lite:special_rules.bzl", "nnapi_sl_visibility_allowlist")

_DEFAULT_VISIBILITY = ["//tensorflow/lite:__subpackages__"]

package(
    default_visibility = _DEFAULT_VISIBILITY,
    licenses = ["notice"],
)

cc_library(
    name = "nnapi_support_library_headers",
    hdrs = [
        "include/SupportLibrary.h",
        "include/SupportLibrarySymbols.h",
        "public/NeuralNetworksSupportLibraryImpl.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = _DEFAULT_VISIBILITY + nnapi_sl_visibility_allowlist(),
    deps = [
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/kernels/internal:compatibility",
        "//tensorflow/lite/nnapi:nnapi_lib",
    ],
)

cc_library(
    name = "nnapi_support_library",
    srcs = [
        "SupportLibrary.cc",
    ],
    compatible_with = get_compatible_with_portable(),
    copts = tflite_copts() + tflite_copts_warnings(),
    visibility = _DEFAULT_VISIBILITY + nnapi_sl_visibility_allowlist(),
    deps = [
        ":nnapi_support_library_headers",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/nnapi:nnapi_lib",
    ],
)
