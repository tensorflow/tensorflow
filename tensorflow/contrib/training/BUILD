# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = [
    "//tensorflow:internal",
])

load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")
load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "training_py",
    srcs = [
        "__init__.py",
        "python/__init__.py",
        "python/training/__init__.py",
        "python/training/bucket_ops.py",
        "python/training/device_setter.py",
        "python/training/evaluation.py",
        "python/training/feeding_queue_runner.py",
        "python/training/hparam.py",
        "python/training/resample.py",
        "python/training/sampling_ops.py",
        "python/training/sequence_queueing_state_saver.py",
        "python/training/tensor_queue_dataset.py",
        "python/training/training.py",
        "python/training/tuner.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":protos_all_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:layers_base",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/estimator:inputs_queues",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_test(
    name = "device_setter_test",
    size = "small",
    srcs = ["python/training/device_setter_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":training_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "sequence_queueing_state_saver_test",
    size = "medium",
    srcs = ["python/training/sequence_queueing_state_saver_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":training_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:string_ops",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "batch_sequences_with_states_test",
    size = "medium",
    srcs = ["python/training/batch_sequences_with_states_test.py"],
    srcs_version = "PY2AND3",
    tags = ["manual"],
    deps = [
        ":training_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "feeding_queue_runner_test",
    size = "medium",
    srcs = ["python/training/feeding_queue_runner_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:training",
        "//tensorflow/python/estimator:inputs_queues",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "hparam_test",
    size = "small",
    srcs = ["python/training/hparam_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":training_py",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "resample_test",
    size = "small",
    srcs = ["python/training/resample_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":training_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "sampling_ops_test",
    size = "small",
    srcs = ["python/training/sampling_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":training_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "sampling_ops_threading_test",
    size = "small",
    srcs = ["python/training/sampling_ops_threading_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "notsan",
    ],
    deps = [
        ":training_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "bucket_ops_test",
    size = "medium",
    srcs = ["python/training/bucket_ops_test.py"],
    srcs_version = "PY2AND3",
    tags = ["manual"],
    deps = [
        ":training_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "evaluation_test",
    size = "small",
    srcs = ["python/training/evaluation_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    tags = [
        "manual",
        "notap",  # Disabling until b/33000128 and b/33040312 are fixed.
    ],
    deps = [
        ":training_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/losses:losses_py",
        "//tensorflow/contrib/metrics:metrics_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "training_test",
    size = "large",
    srcs = ["python/training/training_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":training_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/ops/losses",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "tensor_queue_dataset_test",
    size = "large",
    srcs = ["python/training/tensor_queue_dataset_test.py"],
    srcs_version = "PY2AND3",
    tags = ["notsan"],
    deps = [
        ":training_py",
        "//tensorflow/contrib/data/python/kernel_tests:dataset_serialization_test",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/data",
        "//third_party/py/numpy",
    ],
)

tf_proto_library(
    name = "protos_all",
    srcs = glob(["**/*.proto"]),
    cc_api_version = 2,
    java_api_version = 2,
    visibility = ["//visibility:public"],
)
