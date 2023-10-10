# Description:
#   TensorFlow builder (TensorFlow on Demand project).

load("//tensorflow:strict.default.bzl", "py_strict_binary", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "config_detector",
    srcs = ["config_detector.py"],
    data = [
        "//tensorflow/tools/tensorflow_builder/config_detector/data/golden:cuda_cc_golden",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cuda_compute_capability_lib",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_binary(
    name = "cuda_compute_capability",
    srcs = ["data/cuda_compute_capability.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@six_archive//:six",
    ],
)

py_strict_library(
    name = "cuda_compute_capability_lib",
    srcs = ["data/cuda_compute_capability.py"],
    srcs_version = "PY3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@six_archive//:six",
    ],
)
