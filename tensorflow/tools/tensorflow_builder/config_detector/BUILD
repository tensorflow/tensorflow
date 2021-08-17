# Description:
#   TensorFlow builder (TensorFlow on Demand project).

package(
    default_visibility = [
        "//tensorflow/tools/tensorflow_builder/config_detector:__subpackages__",
    ],
    licenses = ["notice"],
)

py_binary(
    name = "config_detector",
    srcs = ["config_detector.py"],
    data = [
        "//tensorflow/tools/tensorflow_builder/config_detector/data/golden:cuda_cc_golden",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cuda_compute_capability",
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
        "@six_archive//:six",
    ],
)

py_binary(
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
