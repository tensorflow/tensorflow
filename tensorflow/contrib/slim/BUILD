# Description:
#   Contains the Slim library, including common neural networks and examples.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "evaluation",
    srcs = ["python/slim/evaluation.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/training:training_py",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "evaluation_test",
    srcs = ["python/slim/evaluation_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":evaluation",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/metrics:metrics_py",
        "//tensorflow/contrib/training:training_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/debug:debug_data",
        "//tensorflow/python/debug:hooks",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "learning",
    srcs = ["python/slim/learning.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/training:training_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:lib",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_test(
    name = "learning_test",
    srcs = ["python/slim/learning_test.py"],
    srcs_version = "PY2AND3",
    tags = ["manual"],
    deps = [
        ":learning",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/losses:losses_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python:session",
        "//tensorflow/python:summary",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
        "//tensorflow/python/debug:debug_data",
        "//tensorflow/python/debug:dumping_wrapper",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "model_analyzer",
    srcs = ["python/slim/model_analyzer.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "nets",
    srcs = ["nets.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/slim/python/slim/nets:alexnet",
        "//tensorflow/contrib/slim/python/slim/nets:inception",
        "//tensorflow/contrib/slim/python/slim/nets:overfeat",
        "//tensorflow/contrib/slim/python/slim/nets:resnet_utils",
        "//tensorflow/contrib/slim/python/slim/nets:resnet_v1",
        "//tensorflow/contrib/slim/python/slim/nets:resnet_v2",
        "//tensorflow/contrib/slim/python/slim/nets:vgg",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "queues",
    srcs = ["python/slim/queues.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "slim",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    visibility = ["//tensorflow:__subpackages__"],
    deps = [
        ":evaluation",
        ":learning",
        ":model_analyzer",
        ":queues",
        ":summaries",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/contrib/losses:losses_py",
        "//tensorflow/contrib/metrics:metrics_py",
        "//tensorflow/contrib/slim/python/slim/data:data_decoder",
        "//tensorflow/contrib/slim/python/slim/data:data_provider",
        "//tensorflow/contrib/slim/python/slim/data:dataset",
        "//tensorflow/contrib/slim/python/slim/data:dataset_data_provider",
        "//tensorflow/contrib/slim/python/slim/data:parallel_reader",
        "//tensorflow/contrib/slim/python/slim/data:prefetch_queue",
        "//tensorflow/contrib/slim/python/slim/data:tfexample_decoder",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "summaries",
    srcs = ["python/slim/summaries.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:summary",
    ],
)

py_test(
    name = "summaries_test",
    srcs = ["python/slim/summaries_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":summaries",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:summary",
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
