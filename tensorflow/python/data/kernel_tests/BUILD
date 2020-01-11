# Tests of TensorFlow kernels written using the Python API.

load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

tf_py_test(
    name = "batch_test",
    size = "small",
    srcs = ["batch_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/ops/ragged",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "cache_test",
    size = "small",
    srcs = ["cache_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:variables",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "checkpoint_test",
    size = "medium",
    srcs = ["checkpoint_test.py"],
    grpc_enabled = True,
    deps = [
        ":test_base",
        "//tensorflow/python:checkpoint_management",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/training/tracking:util",
    ],
)

tf_py_test(
    name = "concatenate_test",
    size = "small",
    srcs = ["concatenate_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "dataset_test",
    size = "small",
    srcs = ["dataset_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "enumerate_test",
    size = "small",
    srcs = ["enumerate_test.py"],
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "filter_test",
    size = "small",
    srcs = ["filter_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "fixed_length_record_dataset_test",
    size = "small",
    srcs = ["fixed_length_record_dataset_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:readers",
    ],
)

tf_py_test(
    name = "flat_map_test",
    size = "medium",
    srcs = ["flat_map_test.py"],
    grpc_enabled = True,
    deps = [
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:training",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/ops/ragged",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "from_generator_test",
    size = "medium",
    srcs = ["from_generator_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "from_sparse_tensor_slices_test",
    size = "small",
    srcs = ["from_sparse_tensor_slices_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "from_tensors_test",
    size = "small",
    srcs = ["from_tensors_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "from_tensor_slices_test",
    size = "small",
    srcs = ["from_tensor_slices_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/ops/ragged",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "interleave_test",
    size = "medium",
    srcs = ["interleave_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "iterator_cluster_test",
    size = "small",
    srcs = ["iterator_cluster_test.py"],
    grpc_enabled = True,
    tags = [
        "no_oss",  # Test flaky due to port collisions.
        "no_windows",
    ],
    deps = [
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:function",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
    ],
)

cuda_py_test(
    name = "iterator_test",
    size = "medium",
    srcs = ["iterator_test.py"],
    grpc_enabled = True,
    deps = [
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:function",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:io_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/data/util:structure",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/training/tracking:util",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "list_files_test",
    size = "small",
    srcs = ["list_files_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "map_test",
    size = "medium",
    srcs = ["map_test.py"],
    shard_count = 4,
    deps = [
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:script_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/ops/ragged",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "multi_device_iterator_test",
    size = "medium",
    srcs = ["multi_device_iterator_test.py"],
    tags = [
        "no_gpu",  #TODO(b/141255188): Enable test after bug is resolved.
        "no_oss",
        "no_windows_gpu",
    ],
    deps = [
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/data/experimental/ops:optimization_options",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:multi_device_iterator_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "memory_cleanup_test",
    size = "medium",
    srcs = ["memory_cleanup_test.py"],
    tags = ["notsan"],  # b/144706539
    deps = [
        ":test_base",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:multi_device_iterator_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "as_numpy_iterator_test",
    size = "small",
    srcs = ["as_numpy_iterator_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor_value",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "optional_test",
    size = "small",
    srcs = ["optional_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:optional_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "options_test",
    size = "small",
    srcs = ["options_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/data/experimental/ops:optimization_options",
        "//tensorflow/python/data/experimental/ops:threading_options",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "padded_batch_test",
    size = "small",
    srcs = ["padded_batch_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "prefetch_test",
    size = "small",
    srcs = ["prefetch_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dataset_ops_gen",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "range_test",
    size = "small",
    srcs = ["range_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

cuda_py_test(
    name = "reduce_test",
    size = "small",
    srcs = ["reduce_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/experimental/ops:testing",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "repeat_test",
    size = "small",
    srcs = ["repeat_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "shard_test",
    size = "small",
    srcs = ["shard_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/ops:dataset_ops",
    ],
)

tf_py_test(
    name = "shuffle_test",
    size = "small",
    srcs = ["shuffle_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:random_seed",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "skip_test",
    size = "small",
    srcs = ["skip_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "take_test",
    size = "small",
    srcs = ["take_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "text_line_dataset_test",
    size = "small",
    srcs = ["text_line_dataset_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:readers",
        "//tensorflow/python/eager:context",
    ],
)

tf_py_test(
    name = "tf_record_dataset_test",
    size = "small",
    srcs = ["tf_record_dataset_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:lib",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:iterator_ops",
        "//tensorflow/python/data/ops:readers",
    ],
)

py_library(
    name = "test_base",
    srcs = ["test_base.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_combinations",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/util:nest",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/ops/ragged",
    ],
)

tf_py_test(
    name = "unbatch_test",
    size = "medium",
    srcs = ["unbatch_test.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/data/kernel_tests:test_base",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/ops/ragged",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "window_test",
    size = "medium",
    srcs = ["window_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "zip_test",
    size = "small",
    srcs = ["zip_test.py"],
    deps = [
        ":test_base",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python/data/ops:dataset_ops",
        "//third_party/py/numpy",
    ],
)
