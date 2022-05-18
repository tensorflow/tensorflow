# Description: Tests defined for Cloud TPUs

load("//tensorflow:tensorflow.bzl", "pytype_library")  # buildifier: disable=same-origin-load
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_test")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
    licenses = ["notice"],
)

pytype_library(
    name = "tpu_embedding_base_test",
    srcs = ["tpu_embedding_base_test.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/saved_model",
        "//tensorflow/python/tpu:tpu_embedding",
        "//tensorflow/python/tpu:tpu_embedding_for_serving",
        "//tensorflow/python/tpu:tpu_embedding_v1",
        "//tensorflow/python/tpu:tpu_embedding_v2",
        "//tensorflow/python/tpu:tpu_strategy_util",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_checkpoint_test",
    srcs = [
        "tpu_embedding_v2_checkpoint_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_optimizer_test",
    srcs = [
        "tpu_embedding_v2_optimizer_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_enqueue_mode_test",
    srcs = [
        "tpu_embedding_v2_enqueue_mode_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_invalid_input_test",
    srcs = [
        "tpu_embedding_v2_invalid_input_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_valid_input_test",
    srcs = [
        "tpu_embedding_v2_valid_input_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_hd_valid_input_test",
    srcs = [
        "tpu_embedding_v2_hd_valid_input_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_hd_invalid_input_test",
    srcs = [
        "tpu_embedding_v2_hd_invalid_input_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_sequence_feature_test",
    srcs = [
        "tpu_embedding_v2_sequence_feature_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_base_test",
    ],
)

pytype_library(
    name = "tpu_embedding_v2_correctness_base_test",
    srcs = ["tpu_embedding_v2_correctness_base_test.py"],
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_correctness_sparse_training_test",
    srcs = [
        "tpu_embedding_v2_correctness_sparse_training_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_correctness_sparse_forward_test",
    srcs = [
        "tpu_embedding_v2_correctness_sparse_forward_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_correctness_ragged_training_test",
    srcs = [
        "tpu_embedding_v2_correctness_ragged_training_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_correctness_ragged_forward_test",
    srcs = [
        "tpu_embedding_v2_correctness_ragged_forward_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_correctness_hd_sparse_training_test",
    srcs = [
        "tpu_embedding_v2_correctness_hd_sparse_training_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_correctness_hd_sparse_forward_test",
    srcs = [
        "tpu_embedding_v2_correctness_hd_sparse_forward_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_correctness_hd_ragged_training_test",
    srcs = [
        "tpu_embedding_v2_correctness_hd_ragged_training_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_correctness_hd_ragged_forward_test",
    srcs = [
        "tpu_embedding_v2_correctness_hd_ragged_forward_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_correctness_dense_lookup_test",
    srcs = [
        "tpu_embedding_v2_correctness_dense_lookup_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v2_correctness_sequence_feature_test",
    srcs = [
        "tpu_embedding_v2_correctness_sequence_feature_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_v2_correctness_base_test",
    ],
)

### tpu embedding v1 tests

tpu_py_test(
    name = "tpu_embedding_v1_checkpoint_test",
    srcs = [
        "tpu_embedding_v1_checkpoint_test.py",
    ],
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":tpu_embedding_base_test",
    ],
)

tpu_py_test(
    name = "tpu_embedding_v1_correctness_test",
    srcs = [
        "tpu_embedding_v1_correctness_test.py",
    ],
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_oss"],
    deps = [
        ":tpu_embedding_base_test",
    ],
)
