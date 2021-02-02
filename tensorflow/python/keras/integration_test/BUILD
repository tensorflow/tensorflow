# Description:
#   Contains Keras integration tests that verify with other TF high level APIs.

load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_py_test")  # buildifier: disable=same-origin-load
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_test")

package(
    default_visibility = [
        "//tensorflow/tools/pip_package:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

tf_py_test(
    name = "forwardprop_test",
    srcs = ["forwardprop_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "function_test",
    srcs = ["function_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "gradients_test",
    srcs = ["gradients_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "legacy_rnn_test",  # Remove this target in when TF 1 is deprecated.
    srcs = ["legacy_rnn_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "module_test",
    srcs = ["module_test.py"],
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tf_py_test(
    name = "vectorized_map_test",
    srcs = ["vectorized_map_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

cuda_py_test(
    name = "gradient_checkpoint_test",
    srcs = ["gradient_checkpoint_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

tpu_py_test(
    name = "tpu_strategy_test",
    srcs = ["tpu_strategy_test.py"],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    tags = ["no_oss"],
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "multi_worker_tutorial_test",
    srcs = ["multi_worker_tutorial_test.py"],
    python_version = "PY3",
    shard_count = 3,
    tags = [
        "no_tfrt",  # TODO(b/171765113)
        "noasan",  # TODO(b/156029134)
        "nomsan",  # TODO(b/156029134)
        "notsan",  # TODO(b/156029134)
    ],
    deps = [
        "//tensorflow:tensorflow_py",
        "@absl_py//absl/testing:parameterized",
    ],
)
