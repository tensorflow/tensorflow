load("//tensorflow:tensorflow.default.bzl", "get_compatible_with_portable")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "quantization_utils",
    srcs = ["quantization_utils.cc"],
    hdrs = ["quantization_utils.h"],
    compatible_with = get_compatible_with_portable(),
    deps = [
        "//tensorflow/compiler/mlir/tools:safe_cast",
        "//tensorflow/core/framework:tensor_shape",
        "//tensorflow/core/platform:logging",
        "//tensorflow/core/platform:macros",
        "@com_google_absl//absl/types:span",
        "@local_xla//xla/tsl/platform:status",
    ],
)
