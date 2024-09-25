load("//tensorflow:pytype.default.bzl", "pytype_strict_library")
load("//tensorflow:strict.default.bzl", "py_strict_library")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test", "tf_py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cuda_py_strict_test(
    name = "cross_device_ops_test",
    srcs = ["cross_device_ops_test.py"],
    python_version = "PY3",
    tags = [
        # "multi_and_single_gpu", # TODO(b/287692888): re-enable once the 2gpu test passes.
        "no_windows_gpu",  # b/216367668
        "no_windows",  # b/326464742
    ],
    deps = [
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/distribute:collective_all_reduce_strategy",
        "//tensorflow/python/distribute:collective_util",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:cross_device_ops",
        "//tensorflow/python/distribute:cross_device_utils",
        "//tensorflow/python/distribute:device_util",
        "//tensorflow/python/distribute:distribute_utils",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute:multi_worker_util",
        "//tensorflow/python/distribute:reduce_util",
        "//tensorflow/python/distribute:strategy_combinations",
        "//tensorflow/python/distribute:values",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:kernels",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:collective_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "all_reduce",
    srcs = [
        "all_reduce.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:device",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nccl_ops",
    ],
)

tf_py_strict_test(
    name = "all_reduce_test",
    srcs = ["all_reduce_test.py"],
    deps = [
        ":all_reduce",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:state_ops",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:tf_logging",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "input_lib",
    srcs = ["input_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/data/experimental/ops:cardinality",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:multi_device_iterator_ops",
        "//tensorflow/python/data/ops:optional_ops",
        "//tensorflow/python/distribute:input_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/types:data",
        "//tensorflow/python/util:deprecation",
    ],
)
