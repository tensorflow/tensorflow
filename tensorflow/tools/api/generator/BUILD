# Description:
# Scripts used to generate TensorFlow Python API.
licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

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

py_binary(
    name = "create_python_api",
    srcs = ["create_python_api.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "create_python_api_test",
    srcs = ["create_python_api_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":create_python_api",
        "//tensorflow/python:client_testlib",
    ],
)

genrule(
    name = "python_api_gen",
    # List of API files. This list should include file name for
    # every module exported using tf_export. For e.g. if an op is decorated with
    # @tf_export('module1.module2', 'module3'). Then, outs should include
    # api/module1/module2/__init__.py and api/module3/__init__.py.
    outs = [
        "api/__init__.py",
        "api/bitwise/__init__.py",
        "api/contrib/__init__.py",
        "api/contrib/stat_summarizer/__init__.py",
        "api/distributions/__init__.py",
        "api/distributions/bijectors/__init__.py",
        "api/errors/__init__.py",
        "api/image/__init__.py",
        "api/linalg/__init__.py",
        "api/nn/__init__.py",
        "api/spectral/__init__.py",
        "api/train/__init__.py",
        "api/app/__init__.py",
        "api/gfile/__init__.py",
        "api/graph_util/__init__.py",
        "api/keras/__init__.py",
        "api/keras/backend/__init__.py",
        "api/keras/datasets/__init__.py",
        "api/keras/datasets/boston_housing/__init__.py",
        "api/keras/datasets/cifar10/__init__.py",
        "api/keras/datasets/cifar100/__init__.py",
        "api/keras/datasets/imdb/__init__.py",
        "api/keras/datasets/mnist/__init__.py",
        "api/keras/datasets/reuters/__init__.py",
        "api/keras/utils/__init__.py",
        "api/logging/__init__.py",
        "api/resource_loader/__init__.py",
        "api/sysconfig/__init__.py",
        "api/test/__init__.py",
        "api/initializers/__init__.py",
        "api/keras/initializers/__init__.py",
        "api/metrics/__init__.py",
        "api/nn/rnn_cell/__init__.py",
        "api/sets/__init__.py",
        "api/summary/__init__.py",
    ],
    cmd = "$(location create_python_api) $(OUTS)",
    tools = ["create_python_api"],
)

py_library(
    name = "python_api",
    srcs = [":python_api_gen"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib:contrib_py",  # keep
    ],
)
