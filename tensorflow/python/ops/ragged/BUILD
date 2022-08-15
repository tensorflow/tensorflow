load("//tensorflow:tensorflow.bzl", "py_strict_test")
load("//tensorflow:tensorflow.bzl", "py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow/tools/test:performance.bzl", "tf_py_logged_benchmark")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

#-------------------------------------------------------------------------------
# RaggedTensor
#-------------------------------------------------------------------------------

py_library(
    name = "ragged",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    tags = ["nofixdeps"],
    deps = [
        ":dynamic_ragged_shape",
        ":ragged_array_ops",
        ":ragged_batch_gather_ops",
        ":ragged_batch_gather_with_default_op",
        ":ragged_check_ops",
        ":ragged_concat_ops",
        ":ragged_config",
        ":ragged_conversion_ops",
        ":ragged_dispatch",
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_gather_ops",
        ":ragged_getitem",
        ":ragged_image_ops",
        ":ragged_map_ops",
        ":ragged_math_ops",
        ":ragged_operators",
        ":ragged_ops",
        ":ragged_string_ops",
        ":ragged_tensor",
        ":ragged_tensor_shape",
        ":ragged_tensor_value",
        ":ragged_util",
        ":ragged_where_op",
        ":segment_id_ops",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "ragged_array_ops",
    srcs = ["ragged_array_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":dynamic_ragged_shape",
        ":ragged_functional_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        ":ragged_util",
        ":segment_id_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ragged_array_ops_gen",
        "//tensorflow/python:sort_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "ragged_check_ops",
    srcs = ["ragged_check_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        "//tensorflow/python:check_ops",
    ],
)

py_library(
    name = "ragged_batch_gather_ops",
    srcs = ["ragged_batch_gather_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_gather_ops",
        ":ragged_tensor",
        ":ragged_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
    ],
)

py_library(
    name = "ragged_batch_gather_with_default_op",
    srcs = [
        "ragged_batch_gather_with_default_op.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_dispatch",
        ":ragged_operators",
        ":ragged_tensor",
        ":ragged_tensor_shape",
        ":ragged_where_op",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
    ],
)

py_library(
    name = "ragged_concat_ops",
    srcs = ["ragged_concat_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_gather_ops",
        ":ragged_tensor",
        ":ragged_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
    ],
)

py_library(
    name = "ragged_conversion_ops",
    srcs = ["ragged_conversion_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        "//tensorflow/python:dtypes",
    ],
)

py_library(
    name = "ragged_factory_ops",
    srcs = ["ragged_factory_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        ":ragged_tensor_value",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "ragged_functional_ops",
    srcs = ["ragged_functional_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_config",
        ":ragged_tensor",
        ":ragged_util",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "ragged_gather_ops",
    srcs = ["ragged_gather_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ragged_array_ops_gen",
        "//tensorflow/python:tensor_shape",
    ],
)

py_library(
    name = "ragged_getitem",
    srcs = ["ragged_getitem.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_gather_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "ragged_math_ops",
    srcs = ["ragged_math_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_functional_ops",
        ":ragged_tensor",
        ":ragged_util",
        ":segment_id_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ragged_math_ops_gen",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "ragged_operators",
    srcs = ["ragged_operators.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_getitem",
        ":ragged_tensor",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "ragged_ops",
    srcs = ["ragged_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":dynamic_ragged_shape",
        ":ragged_array_ops",
        ":ragged_batch_gather_ops",
        ":ragged_batch_gather_with_default_op",
        ":ragged_check_ops",
        ":ragged_concat_ops",
        ":ragged_config",  # fixdeps: keep
        ":ragged_conversion_ops",
        ":ragged_dispatch",
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_gather_ops",
        ":ragged_getitem",
        ":ragged_image_ops",
        ":ragged_map_ops",
        ":ragged_math_ops",
        ":ragged_operators",
        ":ragged_squeeze_op",
        ":ragged_string_ops",
        ":ragged_tensor",
        ":ragged_tensor_shape",
        ":ragged_tensor_value",
        ":ragged_where_op",
        ":segment_id_ops",
    ],
)

py_library(
    name = "ragged_string_ops",
    srcs = ["ragged_string_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "ragged_squeeze_op",
    srcs = ["ragged_squeeze_op.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        ":ragged_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
    ],
)

py_library(
    name = "ragged_config",
    srcs = ["ragged_config.py"],
    srcs_version = "PY3",
)

py_library(
    name = "row_partition",
    srcs = ["row_partition.py"],
    srcs_version = "PY3",
    deps = [
        ":segment_id_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:bincount_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "ragged_tensor",
    srcs = ["ragged_tensor.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_config",
        ":ragged_tensor_value",
        ":ragged_util",
        ":row_partition",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ragged_conversion_ops_gen",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:tf2",
        "//tensorflow/python:type_spec",
        "//tensorflow/python:util",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:composite_tensor_gradient",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "ragged_tensor_shape",
    srcs = ["ragged_tensor_shape.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_config",
        ":ragged_tensor",
        ":ragged_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
    ],
)

py_library(
    name = "dynamic_ragged_shape",
    srcs = ["dynamic_ragged_shape.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        ":row_partition",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/types",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "ragged_tensor_value",
    srcs = ["ragged_tensor_value.py"],
    srcs_version = "PY3",
    deps = [
        ":row_partition",
        "//tensorflow/python:util",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "ragged_util",
    srcs = ["ragged_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ragged_math_ops_gen",
    ],
)

py_library(
    name = "ragged_where_op",
    srcs = ["ragged_where_op.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_concat_ops",
        ":ragged_functional_ops",
        ":ragged_gather_ops",
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
    ],
)

py_library(
    name = "segment_id_ops",
    srcs = ["segment_id_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:bincount_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "ragged_image_ops",
    srcs = ["ragged_image_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:map_fn",
        "//tensorflow/python/framework:tensor_spec",
    ],
)

py_library(
    name = "ragged_map_ops",
    srcs = ["ragged_map_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_config",
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_array_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/eager:context",
    ],
)

py_library(
    name = "ragged_dispatch",
    srcs = ["ragged_dispatch.py"],
    srcs_version = "PY3",
    deps = [
        ":dynamic_ragged_shape",  # fixdeps: keep
        ":ragged_array_ops",  # fixdeps: keep
        ":ragged_batch_gather_ops",  # fixdeps: keep
        ":ragged_concat_ops",  # fixdeps: keep
        ":ragged_gather_ops",  # fixdeps: keep
        ":ragged_math_ops",  # fixdeps: keep
        ":ragged_squeeze_op",  # fixdeps: keep
        ":ragged_tensor",
        ":ragged_tensor_shape",
        ":ragged_util",  # fixdeps: keep
        ":ragged_where_op",  # fixdeps: keep
        "//tensorflow/python:array_ops",  # fixdeps: keep
        "//tensorflow/python:bitwise_ops",  # fixdeps: keep
        "//tensorflow/python:clip_ops",  # fixdeps: keep
        "//tensorflow/python:dtypes",  # fixdeps: keep
        "//tensorflow/python:framework_ops",  # fixdeps: keep
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",  # fixdeps: keep
        "//tensorflow/python:sparse_tensor",  # fixdeps: keep
        "//tensorflow/python:string_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",  # fixdeps: keep
        "//tensorflow/python/util:tf_decorator",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",  # fixdeps: keep
    ],
)

py_library(
    name = "ragged_tensor_test_ops",
    srcs = ["ragged_tensor_test_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:bitwise_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:string_ops",
    ],
)

#-------------------------------------------------------------------------------
# RaggedTensor Tests
#-------------------------------------------------------------------------------

py_test(
    name = "ragged_tensor_test",
    srcs = ["ragged_tensor_test.py"],
    python_version = "PY3",
    shard_count = 8,
    srcs_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        ":ragged_tensor_value",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:tensor_array_grad",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:type_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_math_ops_test",
    size = "medium",
    srcs = ["ragged_math_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_math_ops",
        "//tensorflow/python:framework_test_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_getitem_test",
    size = "medium",
    timeout = "long",
    srcs = ["ragged_getitem_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        ":ragged_tensor_value",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "row_partition_test",
    srcs = ["row_partition_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":ragged",  # fixdeps: keep
        ":row_partition",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_eager_test",
    size = "medium",
    srcs = ["ragged_eager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_range_op_test",
    srcs = ["ragged_range_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_math_ops",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "ragged_tensor_bounding_shape_op_test",
    srcs = ["ragged_tensor_bounding_shape_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "ragged_row_lengths_op_test",
    srcs = ["ragged_row_lengths_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_gather_op_test",
    srcs = ["ragged_gather_op_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    tags = [
        "notsan",  # TODO(b/170902201): Flaky
    ],
    deps = [
        ":ragged_factory_ops",
        ":ragged_gather_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:context",
    ],
)

py_test(
    name = "ragged_batch_gather_op_test",
    srcs = ["ragged_batch_gather_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_batch_gather_ops",
        ":ragged_batch_gather_with_default_op",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:context",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_gather_nd_op_test",
    srcs = ["ragged_gather_nd_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_gather_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_row_splits_to_segment_ids_op_test",
    srcs = ["ragged_row_splits_to_segment_ids_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":segment_id_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "ragged_segment_ids_to_row_splits_op_test",
    srcs = ["ragged_segment_ids_to_row_splits_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":segment_id_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "ragged_from_tensor_op_test",
    srcs = ["ragged_from_tensor_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_to_sparse_op_test",
    srcs = ["ragged_to_sparse_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:context",
    ],
)

py_test(
    name = "ragged_from_sparse_op_test",
    srcs = ["ragged_from_sparse_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python/eager:context",
    ],
)

py_test(
    name = "ragged_to_tensor_op_test",
    srcs = ["ragged_to_tensor_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_logged_benchmark(
    name = "ragged_to_tensor_op_benchmark",
    target = "//tensorflow/python/ops/ragged:ragged_to_tensor_op_test",
)

py_test(
    name = "ragged_segment_op_test",
    srcs = ["ragged_segment_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_reduce_op_test",
    srcs = ["ragged_reduce_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_math_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_map_flat_values_op_test",
    srcs = ["ragged_map_flat_values_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "ragged_const_op_test",
    srcs = ["ragged_const_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "strings_reduce_join_op_test",
    srcs = ["strings_reduce_join_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged",
        ":ragged_factory_ops",
        ":ragged_string_ops",
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:tensor_shape",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_constant_value_op_test",
    srcs = ["ragged_constant_value_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor_value",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "convert_to_tensor_or_ragged_tensor_op_test",
    srcs = ["convert_to_tensor_or_ragged_tensor_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_boolean_mask_op_test",
    srcs = ["ragged_boolean_mask_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:context",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_concat_op_test",
    srcs = ["ragged_concat_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_concat_ops",
        ":ragged_factory_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:context",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_stack_op_test",
    srcs = ["ragged_stack_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_concat_ops",
        ":ragged_factory_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_rank_op_test",
    srcs = ["ragged_rank_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_tile_op_test",
    srcs = ["ragged_tile_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_util_test",
    srcs = ["ragged_util_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_expand_dims_op_test",
    srcs = ["ragged_expand_dims_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_where_op_test",
    srcs = ["ragged_where_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_where_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_dispatch_test",
    srcs = ["ragged_dispatch_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_dispatch",
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_tensor_test_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/util",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_operators_test",
    srcs = ["ragged_operators_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "ragged_resize_image_op_test",
    srcs = ["ragged_resize_image_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged",  # fixdeps: keep
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "ragged_map_fn_op_test",
    size = "small",
    srcs = ["ragged_map_fn_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_map_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_tensor_shape_test",
    srcs = ["ragged_tensor_shape_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_tensor_shape",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "dynamic_ragged_shape_test",
    size = "medium",
    srcs = ["dynamic_ragged_shape_test.py"],
    python_version = "PY3",
    shard_count = 8,
    srcs_version = "PY3",
    deps = [
        ":dynamic_ragged_shape",
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":row_partition",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients_impl",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:tensor_util",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_size_op_test",
    srcs = ["ragged_size_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_placeholder_op_test",
    srcs = ["ragged_placeholder_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:context",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_squeeze_op_test",
    srcs = ["ragged_squeeze_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_conversion_ops",
        ":ragged_factory_ops",
        ":ragged_squeeze_op",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_dynamic_partition_op_test",
    srcs = ["ragged_dynamic_partition_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_merge_dims_op_test",
    srcs = ["ragged_merge_dims_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "string_ngrams_op_test",
    size = "small",
    srcs = ["string_ngrams_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_string_ops",
        "//tensorflow/python:client_testlib",
    ],
)

py_test(
    name = "ragged_reverse_op_test",
    srcs = ["ragged_reverse_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_cross_op_test",
    srcs = ["ragged_cross_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_windows",  # TODO(b/150702952): Reenable windows test once fixed.
    ],
    deps = [
        ":ragged_array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_one_hot_op_test",
    srcs = ["ragged_one_hot_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:tensor_shape",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_print_op_test",
    srcs = ["ragged_print_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_string_ops",
        ":ragged_tensor",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:logging_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python/eager:def_function",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_tensor_supported_values_test",
    srcs = ["ragged_tensor_supported_values_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_tensor_test_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:composite_tensor",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:type_spec",
        "//tensorflow/python:util",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:extension_type",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_matmul_op_test",
    srcs = ["ragged_matmul_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_concat_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_split_op_test",
    srcs = ["ragged_split_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_tensor",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_test(
    name = "ragged_factory_ops_test",
    size = "small",
    srcs = ["ragged_factory_ops_test.py"],
    tags = [
    ],
    deps = [
        ":ragged_functional_ops",
        "//tensorflow/python:map_fn",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/eager:def_function",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_bitcast_op_test",
    srcs = ["ragged_bitcast_op_test.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
