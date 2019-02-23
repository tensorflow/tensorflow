# This directory contains estimators to train and run inference on
# gradient boosted trees on top of TensorFlow.
licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)

load("//tensorflow:tensorflow.bzl", "py_test")

py_library(
    name = "init_py",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":custom_export_strategy",
        ":custom_loss_head",
        ":distillation_loss",
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
        ":estimator_utils",
        ":trainer_hooks",
        "//tensorflow/contrib/boosted_trees:gbdt_batch",
        "//tensorflow/contrib/boosted_trees:model_ops_py",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training_util",
    ],
)

py_library(
    name = "trainer_hooks",
    srcs = ["trainer_hooks.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/learn",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "estimator_utils",
    srcs = ["estimator_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/learn",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
    ],
)

py_test(
    name = "trainer_hooks_test",
    size = "small",
    srcs = ["trainer_hooks_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":trainer_hooks",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:session",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:variables",
    ],
)

py_library(
    name = "custom_loss_head",
    srcs = ["custom_loss_head.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/learn",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
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
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:session",
        "//tensorflow/python/saved_model:loader",
        "//tensorflow/python/saved_model:tag_constants",
    ],
)

py_test(
    name = "custom_export_strategy_test",
    size = "small",
    srcs = ["custom_export_strategy_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":custom_export_strategy",
        "//tensorflow/contrib/boosted_trees/proto:tree_config_proto_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_library(
    name = "estimator",
    srcs = ["estimator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":custom_loss_head",
        ":estimator_utils",
        ":model",
        "//tensorflow/contrib/boosted_trees:losses",
        "//tensorflow/contrib/learn",
        "//tensorflow/python:math_ops",
    ],
)

py_library(
    name = "dnn_tree_combined_estimator",
    srcs = ["dnn_tree_combined_estimator.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":distillation_loss",
        ":estimator_utils",
        ":model",
        ":trainer_hooks",
        "//tensorflow/contrib/boosted_trees:gbdt_batch",
        "//tensorflow/contrib/boosted_trees:model_ops_py",
        "//tensorflow/contrib/learn",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:training",
    ],
)

py_library(
    name = "distillation_loss",
    srcs = ["distillation_loss.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/learn",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
    ],
)

py_test(
    name = "dnn_tree_combined_estimator_test",
    size = "medium",
    timeout = "long",
    srcs = ["dnn_tree_combined_estimator_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_gpu",
        "no_pip_gpu",
        "notsan",
    ],
    deps = [
        ":dnn_tree_combined_estimator",
        "//tensorflow/contrib/boosted_trees:gbdt_batch",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

py_test(
    name = "estimator_test",
    size = "medium",
    srcs = ["estimator_test.py"],
    shard_count = 4,
    srcs_version = "PY2AND3",
    tags = [
        "no_gpu",
        "no_pip_gpu",
        "notsan",
    ],
    deps = [
        ":estimator",
        "//tensorflow/contrib/boosted_trees:gbdt_batch",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)
