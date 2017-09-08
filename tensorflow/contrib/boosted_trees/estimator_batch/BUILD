# This directory contains estimators to train and run inference on
# gradient boosted trees on top of TensorFlow.
licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)

load("//tensorflow:tensorflow.bzl", "py_test")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

py_library(
    name = "init_py",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "custom_export_strategy",
        ":custom_loss_head",
        ":estimator",
        ":model",
        ":trainer_hooks",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/boosted_trees:gbdt_batch",
    ],
)

py_library(
    name = "trainer_hooks",
    srcs = ["trainer_hooks.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/learn",
    ],
)

py_test(
    name = "trainer_hooks_test",
    size = "small",
    srcs = ["trainer_hooks_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":trainer_hooks",
    ],
)

py_library(
    name = "custom_loss_head",
    srcs = ["custom_loss_head.py"],
    srcs_version = "PY2AND3",
    deps = [
    ],
)

py_library(
    name = "custom_export_strategy",
    srcs = ["custom_export_strategy.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/boosted_trees:gbdt_batch",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_py",
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_py",
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_py",
        "//tensorflow/contrib/learn",
    ],
)

py_test(
    name = "custom_export_strategy_test",
    size = "small",
    srcs = ["custom_export_strategy_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "nomac",  # b/63258195
    ],
    deps = [
        ":custom_export_strategy",
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_extensions_py",
        "//tensorflow/contrib/decision_trees/proto:generic_tree_model_py",
    ],
)

py_library(
    name = "estimator",
    srcs = ["estimator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model",
        ":trainer_hooks",
    ],
)
