licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_py_test")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
    ],
)

py_library(
    name = "ipu",
    srcs = [
      "__init__.py",
      "python/autoshard.py",
      "python/ipu_infeed_queue.py",
      "python/loops.py",
      "python/ops.py",
      "python/popnn_normalization.py",
      "python/popnn_rnn.py",
      "python/sharded_optimizer.py",
      "python/utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/plugin/poplar:popnn_ops_py",
        "//tensorflow/compiler/xla:xla_data_proto_py",
        "//tensorflow/contrib/compiler:compiler_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/compiler/plugin/poplar:pop_datastream_ops_py",
    ],
)

tf_py_test(
    name = "ops_test",
    size = "small",
    srcs = ["python/tests/ops_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
    ],
)

tf_py_test(
    name = "while_loop_perf_test",
    size = "small",
    srcs = ["python/tests/while_loop_perf_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "mapping_test",
    size = "small",
    srcs = ["python/tests/mapping_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow/compiler/plugin/poplar:ipu_ops_py",
        "//tensorflow/compiler/tests:xla_test",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "conditional_test",
    size = "small",
    srcs = ["python/tests/conditional_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "resnet18_nobn_test",
    size = "large",
    srcs = ["size_speed_tests/resnet18_nobn_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
    ],
)

tf_py_test(
    name = "lstm_size_test",
    size = "large",
    srcs = ["size_speed_tests/lstm_size_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
    ],
)

tf_py_test(
    name = "lstm_training_test",
    size = "large",
    srcs = ["training_tests/lstm_training_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
    ],
)

tf_py_test(
    name = "while_loop_test",
    size = "medium",
    srcs = ["python/tests/while_loop_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "multi_ipu_test",
    size = "medium",
    tags = ['exclusive'],
    srcs = ["python/tests/multi_ipu_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "sort_op_test",
    size = "medium",
    srcs = ["python/tests/sort_op_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "infeed_outfeed_test",
    size = "medium",
    srcs = ["python/tests/infeed_outfeed_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:variables",
    ],
)

tf_py_test(
    name = "autoshard_test",
    size = "medium",
    srcs = ["python/tests/autoshard_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow:tensorflow_py",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:framework",
        "//tensorflow/python:platform",
    ],
)

tf_py_test(
    name = "popnn_normalization_test",
    size = "small",
    srcs = ["python/tests/popnn_normalization_test.py"],
    additional_deps = [
        ":ipu",
        "//tensorflow/contrib/compiler:xla",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:layers",
    ],
)

test_suite(
    name = "all_tests",
    tests = [
        "autoshard_test",
        "ops_test",
        "while_loop_perf_test",
        "conditional_test",
        "mapping_test",
        "while_loop_test",
        "sort_op_test",
        "multi_ipu_test",
        "infeed_outfeed_test",
        "popnn_normalization_test",
    ],
)

test_suite(
    name = "size_speed_tests",
    tests = [
        "resnet18_nobn_test",
        "lstm_size_test",
    ],
)

test_suite(
    name = "training_tests",
    tests = [
       "lstm_training_test",
    ],
)

test_suite(
    name = "poplar_test_suite",
    tests = [
      ":all_tests",
      "//tensorflow/compiler/tests:poplar_tests",
      "//tensorflow/compiler/xla/tests:poplar_tests",
      "//tensorflow/compiler/plugin/poplar:all_tests",
    ],
)

test_suite(
    name = "poplar_ci_test_suite",
    tests = [
      ":poplar_test_suite",
      ":size_speed_tests",
      ":training_tests",
    ],
)
