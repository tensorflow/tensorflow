load("//tensorflow:tensorflow.bzl", "py_test")

package(
    default_visibility = [
        "//intelligence/datum/prensor:__pkg__",
        "//learning/brain/contrib/text:__pkg__",
        "//nlp/projects/atc/tf/ops:__pkg__",
        "//tensorflow:internal",
    ],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

#-------------------------------------------------------------------------------
# RaggedTensor
#-------------------------------------------------------------------------------

py_library(
    name = "ragged",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    tags = ["nofixdeps"],
    deps = [
        ":ragged_array_ops",
        ":ragged_conversion_ops",
        ":ragged_dispatch",
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_getitem",
        ":ragged_map_ops",
        ":ragged_math_ops",
        ":ragged_operators",
        ":ragged_string_ops",
        ":ragged_tensor",
        ":ragged_tensor_shape",
        ":ragged_tensor_value",
        ":ragged_util",
        ":segment_id_ops",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "ragged_array_ops",
    srcs = ["ragged_array_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_conversion_ops",
        ":ragged_functional_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        ":ragged_util",
        ":segment_id_ops",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ragged_array_ops_gen",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
    ],
)

py_library(
    name = "ragged_conversion_ops",
    srcs = ["ragged_conversion_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:ragged_conversion_ops_gen",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "ragged_factory_ops",
    srcs = ["ragged_factory_ops.py"],
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/ops/ragged:ragged_tensor_value",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "ragged_functional_ops",
    srcs = ["ragged_functional_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "ragged_getitem",
    srcs = ["ragged_getitem.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
    ],
)

py_library(
    name = "ragged_math_ops",
    srcs = ["ragged_math_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_factory_ops",
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
    ],
)

py_library(
    name = "ragged_operators",
    srcs = ["ragged_operators.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_getitem",
        ":ragged_tensor",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "ragged_string_ops",
    srcs = ["ragged_string_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_conversion_ops",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "ragged_tensor",
    srcs = ["ragged_tensor.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_tensor_value",
        ":ragged_util",
        ":segment_id_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:ragged_conversion_ops_gen",
        "//tensorflow/python:session",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "ragged_tensor_shape",
    srcs = ["ragged_tensor_shape.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_conversion_ops",
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_util",
    ],
)

py_library(
    name = "ragged_tensor_value",
    srcs = ["ragged_tensor_value.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "ragged_util",
    srcs = ["ragged_util.py"],
    srcs_version = "PY2AND3",
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
    name = "segment_id_ops",
    srcs = ["segment_id_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "ragged_map_ops",
    srcs = ["ragged_map_ops.py"],
    srcs_version = "PY2AND3",
    deps = [
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
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        ":ragged_tensor_shape",
        ":ragged_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:util",
        "//tensorflow/python:variables",
        "//third_party/py/numpy",
    ],
)

#-------------------------------------------------------------------------------
# RaggedTensor Tests
#-------------------------------------------------------------------------------

py_library(
    name = "ragged_test_util",
    srcs = ["ragged_test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_tensor",
        ":ragged_tensor_value",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//third_party/py/numpy",
    ],
)

py_test(
    name = "ragged_tensor_test",
    size = "medium",
    srcs = ["ragged_tensor_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        ":ragged_tensor_value",
        ":ragged_test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_eager_test",
    size = "medium",
    srcs = ["ragged_eager_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_test_util",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_range_op_test",
    srcs = ["ragged_range_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_math_ops",
        ":ragged_test_util",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "ragged_tensor_bounding_shape_op_test",
    srcs = ["ragged_tensor_bounding_shape_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_test_util",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "ragged_row_lengths_op_test",
    srcs = ["ragged_row_lengths_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_test_util",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_gather_op_test",
    srcs = ["ragged_gather_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_test_util",
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
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_test_util",
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
    name = "ragged_gather_nd_op_test",
    srcs = ["ragged_gather_nd_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_test_util",
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
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_test_util",
        ":segment_id_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "ragged_segment_ids_to_row_splits_op_test",
    srcs = ["ragged_segment_ids_to_row_splits_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_test_util",
        ":segment_id_ops",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "ragged_from_tensor_op_test",
    srcs = ["ragged_from_tensor_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_tensor",
        ":ragged_test_util",
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
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_tensor",
        ":ragged_test_util",
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
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_tensor",
        ":ragged_test_util",
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
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_segment_op_test",
    srcs = ["ragged_segment_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        ":ragged_test_util",
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
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_math_ops",
        ":ragged_test_util",
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
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_tensor",
        ":ragged_test_util",
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
    srcs_version = "PY2AND3",
    deps = [
        ":ragged",
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_test_util",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_constant_value_op_test",
    srcs = ["ragged_constant_value_op_test.py"],
    srcs_version = "PY2AND3",
    tags = [
        "no_windows",
    ],
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_tensor_value",
        ":ragged_test_util",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "convert_to_tensor_or_ragged_tensor_op_test",
    srcs = ["convert_to_tensor_or_ragged_tensor_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_test_util",
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
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_test_util",
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
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_test_util",
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
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_test_util",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_rank_op_test",
    srcs = ["ragged_rank_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_test_util",
        "//tensorflow/python:framework_test_lib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_tile_op_test",
    srcs = ["ragged_tile_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_test_util",
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
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_test_util",
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
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_test_util",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_where_op_test",
    srcs = ["ragged_where_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_test_util",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_dispatch_test",
    srcs = ["ragged_dispatch_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:clip_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_operators_test",
    srcs = ["ragged_operators_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_test_util",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

py_test(
    name = "ragged_map_fn_op_test",
    size = "small",
    srcs = ["ragged_map_fn_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_map_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        ":ragged_test_util",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/keras:backend",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_tensor_shape_test",
    srcs = ["ragged_tensor_shape_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_tensor_shape",
        ":ragged_test_util",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_test(
    name = "ragged_size_op_test",
    srcs = ["ragged_size_op_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_test_util",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "@absl_py//absl/testing:parameterized",
    ],
)
