package(
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow:tensorflow.bzl", "tf_py_test")

tf_kernel_library(
    name = "kafka_kernels",
    srcs = ["kernels/kafka_dataset_ops.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/kernels:bounds_check_lib",
        "//tensorflow/core/kernels:dataset",
        "//third_party/eigen3",
        "@kafka",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["kafka_ops"],
    deps = [
        "//tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_kafka_ops",
    out = "python/ops/gen_kafka_ops.py",
    require_shape_functions = True,
    deps = [":kafka_ops_op_lib"],
)

py_library(
    name = "kafka",
    srcs = [
        "__init__.py",
        "python/ops/kafka_dataset_ops.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":gen_kafka_ops",
        "//tensorflow/contrib/util:util_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:readers",
    ],
)

# The Kafka server has to be setup before running the test.
# The Kafka server is setup through Docker so the Docker engine
# has to be installed.
#
# Once the Docker engine is ready:
# To setup the Kafka server:
# $ bash tensorflow/contrib/kafka/python/kernel_tests/kafka_test.sh start kafka
#
# After the test is complete:
# To team down the Kafka server:
# $ bash tensorflow/contrib/kafka/python/kernel_tests/kafka_test.sh stop kafka
tf_py_test(
    name = "kafka_test",
    srcs = ["python/kernel_tests/kafka_test.py"],
    additional_deps = [
        ":kafka",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
    tags = [
        "manual",
        "notap",
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
