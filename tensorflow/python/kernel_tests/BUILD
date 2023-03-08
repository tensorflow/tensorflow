# Tests of TensorFlow kernels written using the Python API.

load("//tensorflow:tensorflow.default.bzl", "cuda_py_test", "tf_py_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

# CPU-only tests should use tf_py_test, GPU tests use cuda_py_test
# Please avoid the py_tests and cuda_py_tests (plural) while we
# fix the shared/overbroad dependencies.

cuda_py_test(
    name = "benchmark_test",
    size = "small",
    srcs = ["benchmark_test.py"],
    tags = ["no_windows"],
    deps = [
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform",
        "//tensorflow/python:platform_benchmark",
    ],
)

cuda_py_test(
    name = "check_ops_test",
    size = "small",
    srcs = ["check_ops_test.py"],
    xla_tags = [
        "no_cuda_asan",  # times out
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "collective_ops_multi_worker_test",
    size = "medium",
    srcs = ["collective_ops_multi_worker_test.py"],
    python_version = "PY3",
    tags = ["no_rocm"],
    deps = [
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:errors",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:multi_process_runner",
        "//tensorflow/python/distribute:multi_worker_test_base",
        "//tensorflow/python/distribute/cluster_resolver:cluster_resolver_lib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

# b/234468872
cuda_py_test(
    name = "collective_ops_test",
    size = "medium",
    srcs = ["collective_ops_test.py"],
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "no_tfrt",  # TODO(b/185944042)
    ],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:collective_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:test_util",
        "//tensorflow/python/eager:cancellation",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "composite_tensor_ops_test",
    size = "small",
    srcs = ["composite_tensor_ops_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:composite_tensor_ops",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
    ],
)

cuda_py_test(
    name = "critical_section_test",
    size = "medium",
    srcs = ["critical_section_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_assert",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:control_flow_v2_toggles",
        "//tensorflow/python:critical_section_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:while_loop",
        "//tensorflow/python/data/experimental/ops:prefetching_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "garbage_collection_test",
    size = "small",
    srcs = ["garbage_collection_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/eager:context",
    ],
)

cuda_py_test(
    name = "gradient_correctness_test",
    size = "small",
    srcs = ["gradient_correctness_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "logging_ops_logging_level_test",
    size = "small",
    srcs = ["logging_ops_logging_level_test.py"],
    tags = [
        "no_oss",  # b/198486357
        "no_windows",
    ],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:logging_ops",
    ],
)

cuda_py_test(
    name = "logging_ops_test",
    size = "small",
    srcs = ["logging_ops_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_assert",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
    ],
)

tf_py_test(
    name = "metrics_test",
    size = "medium",
    srcs = ["metrics_test.py"],
    shard_count = 20,
    tags = ["no_windows_gpu"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:data_flow_grad",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:metrics",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "numerics_test",
    size = "small",
    srcs = ["numerics_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:numerics",
        "//tensorflow/python:while_loop",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "template_test",
    size = "small",
    srcs = ["template_test.py"],
    deps = [
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_grad",
        "//tensorflow/python:template",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "trace_op_test",
    size = "small",
    srcs = ["trace_op_test.py"],
    tags = ["no_windows_gpu"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:math_ops",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "histogram_ops_test",
    size = "small",
    srcs = ["histogram_ops_test.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:histogram_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:test_lib",
        "//third_party/py/numpy",
    ],
)
