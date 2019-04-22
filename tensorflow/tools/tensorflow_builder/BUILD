# Description:
#   TensorFlow builder (TensorFlow on Demand project).

licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow/tools/tensorflow_builder:__subpackages__",
    ],
)

py_binary(
    name = "config_detector",
    srcs = ["config_detector.py"],
    data = [
        "//tensorflow/tools/tensorflow_builder/data/golden:cuda_cc_golden",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":cuda_compute_capability",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_binary(
    name = "cuda_compute_capability",
    srcs = ["data/cuda_compute_capability.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)
