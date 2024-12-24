load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")
load("//tensorflow:tensorflow.default.bzl", "cuda_py_strict_test")
load("//tensorflow/tools/test:performance.bzl", "tf_py_logged_benchmark")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

#-------------------------------------------------------------------------------
# RaggedTensor
#-------------------------------------------------------------------------------

py_strict_library(
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
        ":ragged_embedding_ops",
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
    ],
)

py_strict_library(
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
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:ragged_array_ops_gen",
        "//tensorflow/python/ops:sort_ops",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "ragged_autograph",
    srcs = ["ragged_autograph.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        "//tensorflow/python/autograph/operators:control_flow",
        "//tensorflow/python/ops:cond",
    ],
)

py_strict_library(
    name = "ragged_check_ops",
    srcs = ["ragged_check_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/util:dispatch",
    ],
)

py_strict_library(
    name = "ragged_batch_gather_ops",
    srcs = ["ragged_batch_gather_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_gather_ops",
        ":ragged_tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/util:dispatch",
    ],
)

py_strict_library(
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
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:math_ops",
    ],
)

py_strict_library(
    name = "ragged_bincount_ops",
    srcs = ["ragged_bincount_ops.py"],
    deps = [
        ":ragged_tensor",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:bincount_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:count_ops_gen",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/util:dispatch",
    ],
)

cuda_py_strict_test(
    name = "ragged_bincount_ops_test",
    size = "small",
    srcs = ["ragged_bincount_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:bincount_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "ragged_concat_ops",
    srcs = ["ragged_concat_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_gather_ops",
        ":ragged_tensor",
        ":ragged_util",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "ragged_conversion_ops",
    srcs = ["ragged_conversion_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:ragged_conversion_ops_gen",
    ],
)

py_strict_library(
    name = "ragged_embedding_ops",
    srcs = ["ragged_embedding_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_functional_ops",
        ":ragged_tensor",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:embedding_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/util:dispatch",
    ],
)

py_strict_library(
    name = "ragged_factory_ops",
    srcs = ["ragged_factory_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        ":ragged_tensor_value",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:numpy_compat",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "ragged_functional_ops",
    srcs = ["ragged_functional_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_config",
        ":ragged_tensor",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "ragged_gather_ops",
    srcs = ["ragged_gather_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:ragged_array_ops_gen",
        "//tensorflow/python/util:dispatch",
    ],
)

py_strict_library(
    name = "ragged_getitem",
    srcs = ["ragged_getitem.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_gather_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "ragged_math_ops",
    srcs = ["ragged_math_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_functional_ops",
        ":ragged_tensor",
        ":segment_id_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:ragged_math_ops_gen",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "ragged_operators",
    srcs = ["ragged_operators.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_getitem",
        ":ragged_tensor",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:tf_decorator_py",
    ],
)

py_strict_library(
    name = "ragged_ops",
    srcs = ["ragged_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_autograph",
        ":ragged_batch_gather_ops",
        ":ragged_batch_gather_with_default_op",
        ":ragged_bincount_ops",
        ":ragged_check_ops",
        ":ragged_concat_ops",
        ":ragged_config",  # fixdeps: keep
        ":ragged_conversion_ops",
        ":ragged_dispatch",
        ":ragged_embedding_ops",
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

py_strict_library(
    name = "ragged_string_ops",
    srcs = ["ragged_string_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_functional_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:string_ops_gen",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "ragged_squeeze_op",
    srcs = ["ragged_squeeze_op.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
    ],
)

py_strict_library(
    name = "ragged_config",
    srcs = ["ragged_config.py"],
    srcs_version = "PY3",
)

py_strict_library(
    name = "row_partition",
    srcs = ["row_partition.py"],
    srcs_version = "PY3",
    deps = [
        ":segment_id_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/framework:type_spec_registry",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:bincount_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:ragged_math_ops_gen",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "ragged_tensor",
    # RaggedTensor must import ragged_ops to ensure that all dispatched
    # ragged ops are registered.
    # Build cycles from this multi-src build target should not occur
    # if proper TF layering is followed, ie ragged_tensor is allowed to depend on
    # other ops but other ops are not allowed to depend on ragged_tensor.
    srcs = [
        "dynamic_ragged_shape.py",
        "ragged_array_ops.py",
        "ragged_autograph.py",
        "ragged_batch_gather_ops.py",
        "ragged_batch_gather_with_default_op.py",
        "ragged_bincount_ops.py",
        "ragged_check_ops.py",
        "ragged_concat_ops.py",
        "ragged_conversion_ops.py",
        "ragged_dispatch.py",
        "ragged_embedding_ops.py",
        "ragged_factory_ops.py",
        "ragged_functional_ops.py",
        "ragged_gather_ops.py",
        "ragged_getitem.py",
        "ragged_image_ops.py",
        "ragged_map_ops.py",
        "ragged_math_ops.py",
        "ragged_operators.py",
        "ragged_ops.py",
        "ragged_squeeze_op.py",
        "ragged_string_ops.py",
        "ragged_tensor.py",
        "ragged_tensor_shape.py",
        "ragged_where_op.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":ragged_config",
        ":ragged_tensor_value",
        ":ragged_util",
        ":row_partition",
        ":segment_id_ops",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/autograph/operators:control_flow",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:composite_tensor_gradient",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/framework:type_spec_registry",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:bincount_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:count_ops_gen",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:embedding_ops",
        "//tensorflow/python/ops:image_ops",
        "//tensorflow/python/ops:logging_ops",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:ragged_array_ops_gen",
        "//tensorflow/python/ops:ragged_conversion_ops_gen",
        "//tensorflow/python/ops:ragged_math_ops_gen",
        "//tensorflow/python/ops:sort_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:string_ops_gen",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/types:core",
        "//tensorflow/python/types:internal",
        "//tensorflow/python/util:compat",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:numpy_compat",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//tensorflow/tools/docs:doc_controls",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "ragged_tensor_shape",
    srcs = ["ragged_tensor_shape.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_config",
        ":ragged_tensor",
        ":ragged_util",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:math_ops",
    ],
)

py_strict_library(
    name = "dynamic_ragged_shape",
    srcs = ["dynamic_ragged_shape.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        ":row_partition",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/types:core",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "ragged_tensor_value",
    srcs = ["ragged_tensor_value.py"],
    srcs_version = "PY3",
    deps = [
        ":row_partition",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "ragged_util",
    srcs = ["ragged_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:ragged_math_ops_gen",
    ],
)

py_strict_library(
    name = "ragged_where_op",
    srcs = ["ragged_where_op.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_concat_ops",
        ":ragged_functional_ops",
        ":ragged_gather_ops",
        ":ragged_tensor",
        ":ragged_tensor_shape",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:dispatch",
    ],
)

py_strict_library(
    name = "segment_id_ops",
    srcs = ["segment_id_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_util",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:bincount_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "ragged_image_ops",
    srcs = ["ragged_image_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:image_ops",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:dispatch",
    ],
)

py_strict_library(
    name = "ragged_map_ops",
    srcs = ["ragged_map_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/util:nest",
    ],
)

py_strict_library(
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
        "//tensorflow/python/framework:dtypes",  # fixdeps: keep
        "//tensorflow/python/framework:ops",  # fixdeps: keep
        "//tensorflow/python/framework:sparse_tensor",  # fixdeps: keep
        "//tensorflow/python/ops:array_ops",  # fixdeps: keep
        "//tensorflow/python/ops:bitwise_ops",  # fixdeps: keep
        "//tensorflow/python/ops:clip_ops",  # fixdeps: keep
        "//tensorflow/python/ops:logging_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:parsing_ops",  # fixdeps: keep
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:variables",  # fixdeps: keep
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_decorator_py",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:tf_inspect",
        "//third_party/py/numpy",  # fixdeps: keep
    ],
)

py_strict_library(
    name = "ragged_tensor_test_ops",
    srcs = ["ragged_tensor_test_ops.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:bitwise_ops_gen",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_impl",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:parsing_ops",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/ops:string_ops",
    ],
)

#-------------------------------------------------------------------------------
# RaggedTensor Tests
#-------------------------------------------------------------------------------

py_strict_test(
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
        ":ragged_concat_ops",
        ":ragged_factory_ops",
        ":ragged_gather_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        ":ragged_tensor_value",
        ":row_partition",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/framework:type_utils",
        "//tensorflow/python/ops:array_grad",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:math_grad",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:ragged_conversion_ops_gen",
        "//tensorflow/python/ops:tensor_array_grad",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_math_ops_test",
    size = "medium",
    srcs = ["ragged_math_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_math_ops",
        ":ragged_string_ops",
        ":ragged_tensor",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
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
        ":ragged_tensor",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
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
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_eager_test",
    size = "medium",
    srcs = ["ragged_eager_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_range_op_test",
    srcs = ["ragged_range_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_math_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:ragged_math_ops_gen",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "ragged_tensor_bounding_shape_op_test",
    srcs = ["ragged_tensor_bounding_shape_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_row_lengths_op_test",
    srcs = ["ragged_row_lengths_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_gather_op_test",
    srcs = ["ragged_gather_op_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_gather_ops",
        ":ragged_tensor",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_batch_gather_op_test",
    srcs = ["ragged_batch_gather_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_batch_gather_ops",
        ":ragged_batch_gather_with_default_op",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_gather_nd_op_test",
    srcs = ["ragged_gather_nd_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_gather_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_row_splits_to_segment_ids_op_test",
    srcs = ["ragged_row_splits_to_segment_ids_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":segment_id_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
    ],
)

py_strict_test(
    name = "ragged_segment_ids_to_row_splits_op_test",
    srcs = ["ragged_segment_ids_to_row_splits_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":segment_id_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
    ],
)

py_strict_test(
    name = "ragged_from_tensor_op_test",
    srcs = ["ragged_from_tensor_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
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
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:test",
    ],
)

py_strict_test(
    name = "ragged_from_sparse_op_test",
    srcs = ["ragged_from_sparse_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_tensor",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
    ],
)

py_strict_test(
    name = "ragged_to_tensor_op_test",
    srcs = ["ragged_to_tensor_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python/client:session",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/platform:benchmark",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_logged_benchmark(
    name = "ragged_to_tensor_op_benchmark",
    target = "//tensorflow/python/ops/ragged:ragged_to_tensor_op_test",
)

py_strict_test(
    name = "ragged_segment_op_test",
    srcs = ["ragged_segment_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_reduce_op_test",
    srcs = ["ragged_reduce_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_math_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_map_flat_values_op_test",
    srcs = ["ragged_map_flat_values_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_functional_ops",
        ":ragged_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:test",
    ],
)

py_strict_test(
    name = "ragged_const_op_test",
    srcs = ["ragged_const_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "strings_reduce_join_op_test",
    srcs = ["strings_reduce_join_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_string_ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
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
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "convert_to_tensor_or_ragged_tensor_op_test",
    srcs = ["convert_to_tensor_or_ragged_tensor_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_boolean_mask_op_test",
    srcs = ["ragged_boolean_mask_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_concat_op_test",
    srcs = ["ragged_concat_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_concat_ops",
        ":ragged_factory_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_stack_op_test",
    srcs = ["ragged_stack_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_concat_ops",
        ":ragged_factory_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_rank_op_test",
    srcs = ["ragged_rank_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_tile_op_test",
    srcs = ["ragged_tile_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_util_test",
    srcs = ["ragged_util_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_util",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_expand_dims_op_test",
    srcs = ["ragged_expand_dims_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_where_op_test",
    srcs = ["ragged_where_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_where_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_dispatch_test",
    srcs = ["ragged_dispatch_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dynamic_ragged_shape",
        ":ragged",  # fixdeps: keep
        ":ragged_dispatch",
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_tensor_test_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:clip_ops",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/ops:image_ops_impl",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_impl",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_operators_test",
    srcs = ["ragged_operators_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        "//tensorflow/python:tf2",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
    ],
)

py_strict_test(
    name = "ragged_resize_image_op_test",
    srcs = ["ragged_resize_image_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_concat_ops",
        ":ragged_tensor",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:image_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
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
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_tensor_shape_test",
    srcs = ["ragged_tensor_shape_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_tensor_shape",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "dynamic_ragged_shape_test",
    size = "medium",
    srcs = ["dynamic_ragged_shape_test.py"],
    python_version = "PY3",
    shard_count = 8,
    srcs_version = "PY3",
    deps = [
        ":dynamic_ragged_shape",
        ":ragged",  # fixdeps: keep
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":row_partition",
        "//tensorflow/python/client:session",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:math_ops_gen",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_size_op_test",
    srcs = ["ragged_size_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_placeholder_op_test",
    srcs = ["ragged_placeholder_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_squeeze_op_test",
    srcs = ["ragged_squeeze_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_conversion_ops",
        ":ragged_factory_ops",
        ":ragged_squeeze_op",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

cuda_py_strict_test(
    name = "ragged_dynamic_partition_op_test",
    srcs = ["ragged_dynamic_partition_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_concat_ops",
        ":ragged_factory_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:data_flow_ops",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_merge_dims_op_test",
    srcs = ["ragged_merge_dims_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "string_ngrams_op_test",
    size = "small",
    srcs = ["string_ngrams_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_string_ops",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_reverse_op_test",
    srcs = ["ragged_reverse_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_cross_op_test",
    srcs = ["ragged_cross_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "no_windows",  # TODO(b/150702952): Reenable windows test once fixed.
    ],
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:sparse_tensor",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:ragged_array_ops_gen",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_one_hot_op_test",
    srcs = ["ragged_one_hot_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_print_op_test",
    srcs = ["ragged_print_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged",  # fixdeps: keep
        ":ragged_factory_ops",
        ":ragged_string_ops",
        ":ragged_tensor",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:logging_ops",
        "//tensorflow/python/ops:sparse_ops",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_tensor_supported_values_test",
    srcs = ["ragged_tensor_supported_values_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":ragged_factory_ops",
        ":ragged_tensor",
        ":ragged_tensor_test_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:extension_type",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:clip_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/platform:test",
        "//tensorflow/python/util:dispatch",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_matmul_op_test",
    srcs = ["ragged_matmul_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_concat_ops",
        ":ragged_factory_ops",
        ":ragged_math_ops",
        ":ragged_tensor",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_split_op_test",
    srcs = ["ragged_split_op_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ragged_array_ops",
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

cuda_py_strict_test(
    name = "ragged_factory_ops_test",
    srcs = ["ragged_factory_ops_test.py"],
    deps = [
        ":ragged_factory_ops",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/data/ops:multi_device_iterator_ops",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:string_ops",
        "//tensorflow/python/platform:client_testlib",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_bitcast_op_test",
    srcs = ["ragged_bitcast_op_test.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/platform:test",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "ragged_fill_empty_rows_op_test",
    srcs = ["ragged_fill_empty_rows_op_test.py"],
    deps = [
        ":ragged_array_ops",
        ":ragged_factory_ops",
        ":ragged_tensor",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/ops:gradient_checker",
        "//tensorflow/python/platform:test",
        "//third_party/py/numpy",
    ],
)
