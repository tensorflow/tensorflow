# Description:
#   Contains the Keras preprocess layers (internal TensorFlow version).

load("//tensorflow:tensorflow.bzl", "tf_py_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow/core/platform/default:distribute.bzl", "distribute_py_test")

package(
    default_visibility = [
        "//tensorflow/python/keras:__subpackages__",
        "//tensorflow/tools/pip_package:__pkg__",
    ],
    licenses = ["notice"],
)

filegroup(
    name = "all_py_srcs",
    srcs = glob(["*.py"]),
    visibility = ["//tensorflow/python/keras/google/private_tf_api_test:__pkg__"],
)

py_library(
    name = "preprocessing",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":category_crossing",
        ":discretization",
        ":hashing",
        ":image_preprocessing",
        ":integer_lookup",
        ":normalization",
        ":preprocessing_stage",
        ":preprocessing_test_utils",
        ":reduction",
        ":string_lookup",
        ":text_vectorization",
    ],
)

py_library(
    name = "discretization",
    srcs = [
        "discretization.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:boosted_trees_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:resources",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
        "//tensorflow/python/ops/ragged:ragged_functional_ops",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "category_crossing",
    srcs = [
        "category_crossing.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/ops/ragged:ragged_array_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "hashing",
    srcs = [
        "hashing.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "image_preprocessing",
    srcs = [
        "image_preprocessing.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:stateful_random_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:variables",
        "//tensorflow/python/compat",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/preprocessing:image",
        "//tensorflow/python/keras/utils:control_flow_util",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "index_lookup",
    srcs = [
        "index_lookup.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":category_encoding",
        ":table_utils",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/engine",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "normalization",
    srcs = [
        "normalization.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "integer_lookup",
    srcs = [
        "integer_lookup.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":index_lookup",
        ":table_utils",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "table_utils",
    srcs = [
        "table_utils.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/ops/ragged:ragged_functional_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "text_vectorization",
    srcs = [
        "text_vectorization.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":category_encoding",
        ":string_lookup",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/keras/utils:layer_utils",
        "//tensorflow/python/keras/utils:tf_utils",
        "//tensorflow/python/ops/ragged:ragged_functional_ops",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "category_encoding",
    srcs = [
        "category_encoding.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:bincount_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:backend",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/keras/engine:input_spec",
        "//tensorflow/python/keras/utils:layer_utils",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "reduction",
    srcs = [
        "reduction.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python/keras/engine:base_layer",
    ],
)

py_library(
    name = "string_lookup",
    srcs = [
        "string_lookup.py",
    ],
    srcs_version = "PY3",
    deps = [
        ":index_lookup",
        ":table_utils",
        "//tensorflow/python:dtypes",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/util:tf_export",
    ],
)

py_library(
    name = "preprocessing_stage",
    srcs = [
        "preprocessing_stage.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:util",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/keras/utils:tf_utils",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "preprocessing_test_utils",
    srcs = ["preprocessing_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "category_crossing_test",
    srcs = ["category_crossing_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_windows",  # b/149031156
    ],
    deps = [
        ":category_crossing",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:testing_utils",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//tensorflow/python/ops/ragged:ragged_tensor",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "category_encoding_test",
    srcs = ["category_encoding_test.py"],
    python_version = "PY3",
    deps = [
        ":category_encoding",
        ":preprocessing_test_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "category_encoding_distribution_test",
    srcs = ["category_encoding_distribution_test.py"],
    disable_mlir_bridge = False,
    main = "category_encoding_distribution_test.py",
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # b/189866692
        "no_rocm",
        "noguitar",  # b/190034522
    ],
    tpu_tags = [
        "no_oss",  # b/155502591
    ],
    deps = [
        ":category_encoding",
        ":preprocessing_test_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:config",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_combinations_lib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/distribute:strategy_combinations",
    ],
)

distribute_py_test(
    name = "category_crossing_distribution_test",
    srcs = ["category_crossing_distribution_test.py"],
    main = "category_crossing_distribution_test.py",
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
    ],
    tpu_tags = [
        "no_oss",  # b/155502591
    ],
    deps = [
        ":category_crossing",
        ":preprocessing_test_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:config",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_combinations_lib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/distribute:tpu_strategy",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/distribute:strategy_combinations",
    ],
)

distribute_py_test(
    name = "image_preprocessing_distribution_test",
    srcs = ["image_preprocessing_distribution_test.py"],
    main = "image_preprocessing_distribution_test.py",
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "multi_and_single_gpu",
        "no_rocm",
    ],
    tpu_tags = [
        "no_oss",
        "noguitar",  # TODO(b/183957207)
    ],
    deps = [
        ":image_preprocessing",
        ":preprocessing_test_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_combinations_lib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/distribute:strategy_combinations",
    ],
)

tf_py_test(
    name = "discretization_test",
    srcs = ["discretization_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = ["no_rocm"],
    deps = [
        ":discretization",
        ":preprocessing_test_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "discretization_distribution_test",
    srcs = ["discretization_distribution_test.py"],
    main = "discretization_distribution_test.py",
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
        "no_oss",  # TODO(b/189956080)
        "no_rocm",
        "noguitar",  # b/190034522
    ],
    deps = [
        ":discretization",
        ":preprocessing_test_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:config",
        "//tensorflow/python:framework_test_combinations_lib",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/distribute:strategy_combinations",
    ],
)

cuda_py_test(
    name = "hashing_test",
    srcs = ["hashing_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":hashing",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:tensor_spec",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:testing_utils",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/ops/ragged:ragged_factory_ops",
        "//third_party/py/numpy",
    ],
)

distribute_py_test(
    name = "hashing_distribution_test",
    srcs = ["hashing_distribution_test.py"],
    disable_mlir_bridge = False,
    main = "hashing_distribution_test.py",
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
    ],
    deps = [
        ":hashing",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/distribute:strategy_combinations",
    ],
)

tf_py_test(
    name = "index_lookup_test",
    srcs = ["index_lookup_test.py"],
    python_version = "PY3",
    tags = ["noasan"],  # TODO(b/183961255)
    deps = [
        ":index_lookup",
        ":preprocessing_test_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "index_lookup_distribution_test",
    srcs = ["index_lookup_distribution_test.py"],
    disable_mlir_bridge = False,
    main = "index_lookup_distribution_test.py",
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
        "no_rocm",
    ],
    tpu_tags = ["no_oss"],
    deps = [
        ":index_lookup",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/distribute:strategy_combinations",
    ],
)

cuda_py_test(
    name = "image_preprocessing_test",
    srcs = ["image_preprocessing_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "no_windows",  # TODO(b/184424727): Re-enable this.
    ],
    deps = [
        ":image_preprocessing",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:image_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:stateful_random_ops",
        "//tensorflow/python:stateless_random_ops",
        "//tensorflow/python/compat",
        "//tensorflow/python/distribute:mirrored_strategy",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras:testing_utils",
        "//tensorflow/python/keras/engine",
        "//tensorflow/python/keras/utils:generic_utils",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "normalization_test",
    srcs = ["normalization_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = [
        "broken",  # b/170974360
        "noasan",  # TODO(b/337374867) fails with -fsanitize=null
    ],
    deps = [
        ":normalization",
        ":preprocessing_test_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "integer_lookup_test",
    srcs = ["integer_lookup_test.py"],
    python_version = "PY3",
    tags = ["noasan"],  # TODO(b/183961255)
    deps = [
        ":integer_lookup",
        ":preprocessing_test_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "normalization_distribution_test",
    srcs = ["normalization_distribution_test.py"],
    main = "normalization_distribution_test.py",
    python_version = "PY3",
    tags = [
        "no_cuda_asan",
        "no_oss",
    ],
    deps = [
        ":normalization",
        ":preprocessing_test_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_test_combinations_lib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/distribute:strategy_combinations",
    ],
)

tf_py_test(
    name = "table_utils_test",
    srcs = ["table_utils_test.py"],
    python_version = "PY3",
    deps = [
        ":table_utils",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "text_vectorization_test",
    srcs = ["text_vectorization_test.py"],
    python_version = "PY3",
    shard_count = 4,
    deps = [
        ":preprocessing_test_utils",
        ":text_vectorization",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

distribute_py_test(
    name = "text_vectorization_distribution_test",
    srcs = ["text_vectorization_distribution_test.py"],
    main = "text_vectorization_distribution_test.py",
    python_version = "PY3",
    tags = [
        "multi_and_single_gpu",
        "no_rocm",
    ],
    tpu_tags = [
        "no_oss",  # b/155502591
    ],
    deps = [
        ":preprocessing_test_utils",
        ":text_vectorization",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:config",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_test_combinations_lib",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/distribute:combinations",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/distribute:strategy_combinations",
    ],
)

tf_py_test(
    name = "reduction_test",
    srcs = ["reduction_test.py"],
    python_version = "PY3",
    shard_count = 4,
    tags = ["notsan"],  # TODO(b/170783154)
    deps = [
        ":reduction",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "string_lookup_test",
    srcs = ["string_lookup_test.py"],
    python_version = "PY3",
    tags = [
        "notsan",  #b/168758821
    ],
    deps = [
        ":preprocessing_test_utils",
        ":string_lookup",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//tensorflow/python/keras/utils:generic_utils",
        "//tensorflow/python/ops/ragged:ragged_string_ops",
        "@absl_py//absl/testing:parameterized",
    ],
)

tf_py_test(
    name = "preprocessing_stage_test",
    srcs = ["preprocessing_stage_test.py"],
    python_version = "PY3",
    tags = ["no_windows"],  # TODO(b/152991402)
    deps = [
        ":preprocessing_stage",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/keras",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:parameterized",
    ],
)
