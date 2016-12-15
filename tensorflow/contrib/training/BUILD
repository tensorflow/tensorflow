# Description:
#   contains parts of TensorFlow that are experimental or unstable and which are not supported.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

py_library(
    name = "training_py",
    srcs = [
        "__init__.py",
        "python/training/bucket_ops.py",
        "python/training/device_setter.py",
        "python/training/evaluation.py",
        "python/training/failure_tolerator.py",
        "python/training/feeder.py",
        "python/training/resample.py",
        "python/training/sampling_ops.py",
        "python/training/sequence_queueing_state_saver.py",
        "python/training/training.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "device_setter_test",
    size = "small",
    srcs = ["python/training/device_setter_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":training_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "sequence_queueing_state_saver_test",
    size = "medium",
    srcs = ["python/training/sequence_queueing_state_saver_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":training_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
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
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "failure_tolerator_test",
    size = "small",
    srcs = ["python/training/failure_tolerator_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":training_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "feeder_test",
    size = "small",
    srcs = ["python/training/feeder_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":training_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "resample_test",
    size = "small",
    srcs = ["python/training/resample_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":training_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "sampling_ops_test",
    size = "small",
    srcs = ["python/training/sampling_ops_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":training_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform",
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
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
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
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
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
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

py_test(
    name = "training_test",
    size = "large",
    srcs = ["python/training/training_test.py"],
    shard_count = 3,
    srcs_version = "PY2AND3",
    deps = [
        ":training_py",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:framework_test_lib",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
