load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings")

package(
    default_visibility = ["//tensorflow/lite:__subpackages__"],
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
    deps = [
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
    deps = [
        ":nnapi_support_library_headers",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite/nnapi:nnapi_lib",
    ],
)
