package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_library(
    name = "estimator_py",
    srcs = [
        "estimator_lib.py",
    ],
    srcs_version = "PY3",
    visibility = [
        "//tensorflow:__pkg__",
        "//tensorflow:internal",
    ],
    deps = [
        ":baseline",
        ":boosted_trees",
        ":dnn",
        ":dnn_linear_combined",
        ":estimator",
        ":expect_tensorflow_estimator_installed",
        ":export",
        ":exporter",
        ":inputs",
        ":keras",
        ":linear",
        ":model_fn",
        ":parsing_utils",
        ":run_config",
        ":training",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "exporter",
    srcs = ["exporter.py"],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        ":gc",
        ":metric_keys",
        ":util",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "gc",
    srcs = ["gc.py"],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "model_fn",
    srcs = ["model_fn.py"],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        ":export_output",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "training",
    srcs = ["training.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":expect_tensorflow_estimator_installed",
        ":exporter",
        ":run_config",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "run_config",
    srcs = ["run_config.py"],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "baseline",
    srcs = ["canned/baseline.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":expect_tensorflow_estimator_installed",
        ":head",
        ":model_fn",
        ":optimizers",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "boosted_trees",
    srcs = ["canned/boosted_trees.py"],
    srcs_version = "PY3",
    deps = [
        ":boosted_trees_utils",
        ":estimator",
        ":expect_tensorflow_estimator_installed",
        ":head",
        ":model_fn",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "boosted_trees_utils",
    srcs = ["canned/boosted_trees_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":expect_tensorflow_estimator_installed",
        ":head",
        ":model_fn",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "dnn",
    srcs = ["canned/dnn.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":expect_tensorflow_estimator_installed",
        ":head",
        ":model_fn",
        ":optimizers",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "dnn_testing_utils",
    testonly = 1,
    srcs = ["canned/dnn_testing_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":expect_tensorflow_estimator_installed",
        ":head",
        ":metric_keys",
        ":model_fn",
        ":numpy_io",
        ":prediction_keys",
        "//tensorflow:tensorflow_py_no_contrib",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
        "@six_archive//:six",
    ],
)

py_library(
    name = "dnn_linear_combined",
    srcs = ["canned/dnn_linear_combined.py"],
    srcs_version = "PY3",
    deps = [
        ":dnn",
        ":estimator",
        ":expect_tensorflow_estimator_installed",
        ":head",
        ":linear",
        ":model_fn",
        ":optimizers",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "util",
    srcs = [
        "util.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "estimator",
    srcs = [
        "estimator.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        ":export_export",
        ":model_fn",
        ":run_config",
        ":util",
        "//tensorflow:tensorflow_py_no_contrib",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "parsing_utils",
    srcs = [
        "canned/parsing_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "export_output",
    srcs = ["export/export_output.py"],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "export",
    srcs = [
        "export/export_lib.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        ":export_export",
        ":export_output",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "export_export",
    srcs = [
        "export/export.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        ":util",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "head",
    srcs = ["canned/head.py"],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        ":export_output",
        ":metric_keys",
        ":model_fn",
        ":prediction_keys",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "inputs",
    srcs = ["inputs/inputs.py"],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        ":numpy_io",
        ":pandas_io",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "linear",
    srcs = ["canned/linear.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":expect_tensorflow_estimator_installed",
        ":head",
        ":optimizers",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "linear_testing_utils",
    testonly = 1,
    srcs = ["canned/linear_testing_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":expect_tensorflow_estimator_installed",
        ":export_export",
        ":linear",
        ":metric_keys",
        ":numpy_io",
        ":pandas_io",
        ":run_config",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "metric_keys",
    srcs = ["canned/metric_keys.py"],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        ":model_fn",
    ],
)

py_library(
    name = "numpy_io",
    srcs = ["inputs/numpy_io.py"],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        ":inputs_queues",
    ],
)

py_library(
    name = "optimizers",
    srcs = ["canned/optimizers.py"],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "pandas_io",
    srcs = ["inputs/pandas_io.py"],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        ":inputs_queues",
    ],
)

py_library(
    name = "prediction_keys",
    srcs = ["canned/prediction_keys.py"],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
    ],
)

py_library(
    name = "inputs_queues",
    srcs = [
        "inputs/queues/feeding_functions.py",
        "inputs/queues/feeding_queue_runner.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":expect_tensorflow_estimator_installed",
        "//tensorflow:tensorflow_py_no_contrib",
        "@six_archive//:six",
    ],
)

py_library(
    name = "keras",
    srcs = ["keras.py"],
    srcs_version = "PY3",
    deps = [
        ":estimator",
        ":expect_tensorflow_estimator_installed",
        ":export_export",
        ":model_fn",
        ":run_config",
        "//tensorflow:tensorflow_py_no_contrib",
    ],
)

py_library(
    name = "expect_tensorflow_estimator_installed",
    srcs_version = "PY3",
    # This is a dummy rule used as a dependency in open-source.
    # We expect tensorflow_estimator to already be installed.
    visibility = ["//visibility:public"],
)
