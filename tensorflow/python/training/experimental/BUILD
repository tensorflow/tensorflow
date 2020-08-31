load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "loss_scale",
    srcs = ["loss_scale.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "loss_scale_optimizer",
    srcs = ["loss_scale_optimizer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":loss_scale",
        "//tensorflow/python/distribute:distribute_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

# The test currently requires visibility only granted to tensorflow/python:__pkg__
exports_files(
    ["loss_scale_optimizer_test.py"],
    visibility = ["//tensorflow/python:__pkg__"],
)

py_test(
    name = "loss_scale_test",
    size = "medium",
    srcs = ["loss_scale_test.py"],
    python_version = "PY3",
    deps = [
        ":loss_scale",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/distribute:one_device_strategy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "mixed_precision_global_state",
    srcs = ["mixed_precision_global_state.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "mixed_precision",
    srcs = ["mixed_precision.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":loss_scale",
        ":loss_scale_optimizer",
        ":mixed_precision_global_state",
        "//tensorflow/python:config",
        "//tensorflow/python:util",
    ],
)

cuda_py_test(
    name = "mixed_precision_test",
    size = "small",
    srcs = ["mixed_precision_test.py"],
    python_version = "PY3",
    tfrt_enabled = True,
    deps = [
        ":mixed_precision",
        "//tensorflow/python:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_library(
    name = "loss_scaling_gradient_tape",
    srcs = ["loss_scaling_gradient_tape.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":loss_scale",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:unconnected_gradients",
        "//tensorflow/python:util",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:backprop",
    ],
)

cuda_py_test(
    name = "loss_scaling_gradient_tape_test",
    size = "medium",
    srcs = ["loss_scaling_gradient_tape_test.py"],
    shard_count = 2,
    deps = [
        ":loss_scale",
        ":loss_scaling_gradient_tape",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_combinations_lib",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/eager:def_function",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
