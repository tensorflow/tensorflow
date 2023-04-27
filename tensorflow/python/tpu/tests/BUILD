# Description: Tests defined for Cloud TPUs

load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow/python/tpu:tpu.bzl", "tpu_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

pytype_strict_library(
    name = "tpu_embedding_base_test",
    srcs = ["tpu_embedding_base_test.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/eager:remote",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/tpu:tpu_embedding_v2",
        "//tensorflow/python/tpu:tpu_embedding_v2_utils",
        "//tensorflow/python/tpu:tpu_strategy_util",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python/checkpoint:checkpoint_core",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/module",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/tpu:tpu_embedding_for_serving",
        "//tensorflow/python/tpu:tpu_embedding_v2",
        "//tensorflow/python/tpu:tpu_embedding_v2_utils",
        "//tensorflow/python/tpu:tpu_strategy_util",
        "//tensorflow/python/training:checkpoint_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:math_ops_gen",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/tpu:tpu_embedding",
        "//tensorflow/python/tpu:tpu_embedding_v2",
        "//tensorflow/python/tpu:tpu_embedding_v2_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/tpu:tpu_embedding_v2",
        "//tensorflow/python/tpu:tpu_embedding_v2_utils",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/tpu:tpu_embedding_v2",
        "//tensorflow/python/tpu:tpu_embedding_v2_utils",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

pytype_strict_library(
    name = "tpu_embedding_v2_correctness_base_test",
    srcs = ["tpu_embedding_v2_correctness_base_test.py"],
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_base_test",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python:array_ops",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/tpu:tpu_embedding_v2",
        "//tensorflow/python/tpu:tpu_embedding_v2_utils",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_embedding_v2_initialization_test",
    srcs = [
        "tpu_embedding_v2_initialization_test.py",
    ],
    disable_experimental = True,
    disable_mlir_bridge = False,
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":tpu_embedding_base_test",
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/tpu:tpu_embedding_v2",
        "//tensorflow/python/tpu:tpu_embedding_v2_utils",
        "//third_party/py/numpy",
    ],
)

### tpu embedding v1 tests

tpu_py_strict_test(
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
        "//tensorflow/python:init_ops_v2",
        "//tensorflow/python/checkpoint:checkpoint_core",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/tpu:tpu_embedding_for_serving",
        "//tensorflow/python/tpu:tpu_embedding_v1",
        "//tensorflow/python/tpu:tpu_embedding_v2_utils",
        "//tensorflow/python/training:checkpoint_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
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
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/keras/optimizer_v2",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/tpu:tpu_embedding_v1",
        "//tensorflow/python/tpu:tpu_embedding_v2_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tpu_py_strict_test(
    name = "tpu_initialization_test",
    srcs = [
        "tpu_initialization_test.py",
    ],
    disable_mlir_bridge = False,
    disable_tfrt = False,
    disable_v3_4chips = False,
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["no_oss"],
    deps = [
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/distribute/cluster_resolver:tpu_cluster_resolver_py",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/tpu:tpu_strategy_util",
        "@absl_py//absl/testing:parameterized",
    ],
)
