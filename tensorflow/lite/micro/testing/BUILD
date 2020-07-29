load(
    "//tensorflow/lite/micro/testing:micro_test.bzl",
    "tflite_micro_cc_test",
)

licenses(["notice"])  # Apache 2.0

package_group(
    name = "micro",
    packages = ["//tensorflow/lite/micro/..."],
)

package_group(
    name = "microfrontend",
    packages = ["//tensorflow/lite/experimental/microfrontend/..."],
)

exports_files(["test_linux_binary.sh"])

cc_library(
    name = "micro_test",
    srcs = [
        "test_utils.cc",
    ],
    hdrs = [
        "micro_test.h",
        "test_utils.h",
    ],
    visibility = [
        ":micro",
        ":microfrontend",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:test_helpers",
    ],
)

tflite_micro_cc_test(
    name = "util_test",
    srcs = [
        "util_test.cc",
    ],
    deps = [
        ":micro_test",
    ],
)

cc_library(
    name = "test_conv_model",
    srcs = [
        "test_conv_model.cc",
    ],
    hdrs = [
        "test_conv_model.h",
    ],
    visibility = [
        ":micro",
    ],
)

py_binary(
    name = "generate_test_models",
    srcs = ["generate_test_models.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    tags = ["noubsan"],  # TODO(b/144512025): Fix raw_to_bitmap_test to fix ubsan failure.
    deps = [
        "//tensorflow:tensorflow_py",
        "//third_party/py/numpy",
        "@absl_py//absl:app",
    ],
)
