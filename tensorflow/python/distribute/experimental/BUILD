load("//tensorflow:strict.default.bzl", "py_strict_library")
load(
    "//tensorflow/dtensor:build_defs.bzl",
    "GPU_2DEVS_BACKEND",
    "TPU_V3_DONUT_BACKEND",
    "dtensor_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "experimental",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/distribute:central_storage_strategy",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:parameter_server_strategy",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/distribute/failure_handling:failure_handling_lib",
        "//tensorflow/python/distribute/failure_handling:preemption_watcher",
    ],
)

py_strict_library(
    name = "mirrored_strategy",
    srcs = ["mirrored_strategy.py"],
    deps = [
        ":dtensor_strategy_extended",
        ":dtensor_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:mesh_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/distribute:cross_device_ops",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:ops",
    ],
)

dtensor_test(
    name = "mirrored_strategy_test",
    srcs = ["mirrored_strategy_test.py"],
    shard_count = {"tpu": 2},
    tags = ["no_pip"],
    deps = [
        ":dtensor_util",
        ":mirrored_strategy",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:mesh_util",
        "//tensorflow/dtensor/python/tests:test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_spec",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "dtensor_util",
    srcs = ["dtensor_util.py"],
    deps = [
        "//tensorflow/dtensor/python:accelerator_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:input_util",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:summary_ops_v2",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_util",
    ],
)

dtensor_test(
    name = "dtensor_util_test",
    srcs = ["dtensor_util_test.py"],
    tags = ["no_pip"],
    deps = [
        ":dtensor_util",
        ":mirrored_strategy",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python/tests:test_util",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "dtensor_strategy_extended",
    srcs = ["dtensor_strategy_extended.py"],
    deps = [
        ":dtensor_util",
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:input_util",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/python:array_ops",
        "//tensorflow/python/data/experimental/ops:distribute",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:distribute_utils",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/util:nest",
    ],
)

py_strict_library(
    name = "multi_worker_mirrored_strategy",
    srcs = ["multi_worker_mirrored_strategy.py"],
    deps = [
        ":dtensor_strategy_extended",
        ":dtensor_util",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:mesh_util",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute:multi_worker_util",
        "//tensorflow/python/distribute/cluster_resolver:tfconfig_cluster_resolver_py",
    ],
)

dtensor_test(
    name = "multi_worker_mirrored_strategy_test",
    srcs = ["multi_worker_mirrored_strategy_test.py"],
    additional_backends = [
        GPU_2DEVS_BACKEND,
        TPU_V3_DONUT_BACKEND,
    ],
    disable = [
        "gpu",  # multi-client gpu is tested via GPU_2DEVS_BACKEND.
        "tpu",  # multi-client tpu is tested via TPU_V3_DONUT_BACKEND.
    ],
    disable_tfrt = [
        "cpu",  # TODO(b/217969210): Re-enable in TFRT CPU.
        GPU_2DEVS_BACKEND,  # TODO(b/230679405): Re-enable in TFRT GPU.
    ],
    tags = [
        "no_pip",
        "no_windows",
        "nosan",
    ],  # b/195537906
    deps = [
        ":dtensor_util",
        ":multi_worker_mirrored_strategy",
        "//tensorflow/dtensor/python:d_variable",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python/tests:multi_client_test_util",
        "//tensorflow/dtensor/python/tests:test_util",
        "//tensorflow/python:variables",
        "//tensorflow/python/distribute:distribute_lib",
        "//tensorflow/python/distribute/cluster_resolver:tfconfig_cluster_resolver_py",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)
