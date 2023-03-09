package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "distributions",
    srcs = glob(
        ["*.py"],
        exclude = ["util.py"],
    ),
    deprecation = ("TensorFlow Distributions has migrated to " +
                   "TensorFlow Probability " +
                   "(https://github.com/tensorflow/probability). " +
                   "Deprecated copies remaining in tf.distributions " +
                   "will not receive new features, and will be removed by " +
                   "early 2019. You should update all usage of " +
                   "`tf.distributions` to `tfp.distributions`."),
    srcs_version = "PY3",
    deps = [
        ":util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_assert",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:special_math_ops",
        "//tensorflow/python:tensor_util",
        "//third_party/py/numpy",
    ],
)
