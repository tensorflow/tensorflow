load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

licenses(["notice"])

py_strict_library(
    name = "config",
    srcs = ["config.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@absl_py//absl:app",
        "@absl_py//absl/flags",
    ],
)

py_strict_library(
    name = "test_util",
    srcs = ["test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":config",
        ":extensions",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/ops:gradient_checker_v2",
        "//tensorflow/python/ops/numpy_ops:np_array_ops",
        "//tensorflow/python/ops/numpy_ops:np_utils",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_library(
    name = "np_wrapper",
    srcs = ["np_wrapper.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/ops/numpy_ops:np_array_ops",
        "//tensorflow/python/ops/numpy_ops:np_arrays",
        "//tensorflow/python/ops/numpy_ops:np_config",
        "//tensorflow/python/ops/numpy_ops:np_dtypes",
        "//tensorflow/python/ops/numpy_ops:np_math_ops",
        "//tensorflow/python/ops/numpy_ops:np_random",
        "//tensorflow/python/ops/numpy_ops:np_utils",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "extensions",
    srcs = ["extensions.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":np_wrapper",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/compiler/xla",
        "//tensorflow/python/data/ops:dataset_ops",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager/polymorphic_function",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:device_spec",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:indexed_slices",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor",
        "//tensorflow/python/framework:tensor_conversion_registry",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:bitwise_ops_gen",
        "//tensorflow/python/ops:clip_ops",
        "//tensorflow/python/ops:collective_ops_gen",
        "//tensorflow/python/ops:control_flow_assert",
        "//tensorflow/python/ops:custom_gradient",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn_ops",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/ops:tensor_array_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "//tensorflow/python/ops/parallel_for:control_flow_ops",
        "//tensorflow/python/tpu:tpu_py",
        "//tensorflow/python/tpu/ops",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

# copybara:uncomment_begin(google-only)
# py_strict_test(
#     name = "extensions_test",
#     srcs = ["extensions_test.py"],
#     python_version = "PY3",
#     srcs_version = "PY2AND3",
#     tags = [
#         "gpu",
#         "no_pip",
#         "notap",  # b/294137902
#         "requires-gpu-nvidia",
#     ],
#     deps = [
#         ":extensions",
#         ":np_wrapper",
#         "//learning/brain/research/jax:gpu_support",
#         # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
#         "//third_party/py/jax",
#         "//third_party/py/numpy",
#         "//tensorflow:tensorflow_py",
#         "//tensorflow/python/distribute/cluster_resolver/tpu:tpu_cluster_resolver_py",
#         "//tensorflow/python/eager:backprop",
#         "//tensorflow/python/eager:context",
#         "//tensorflow/python/eager/polymorphic_function",
#         "//tensorflow/python/framework:config",
#         "//tensorflow/python/framework:constant_op",
#         "//tensorflow/python/framework:dtypes",
#         "//tensorflow/python/framework:ops",
#         "//tensorflow/python/framework:tensor",
#         "//tensorflow/python/framework:tensor_shape",
#         "//tensorflow/python/ops:array_ops",
#         "//tensorflow/python/ops:gradient_checker_v2",
#         "//tensorflow/python/ops:math_ops",
#         "//tensorflow/python/ops:nn_ops",
#         "//tensorflow/python/ops:random_ops",
#         "//tensorflow/python/ops:stateful_random_ops",
#         "//tensorflow/python/ops/numpy_ops:numpy",
#         "//tensorflow/python/platform:client_testlib",
#         "//tensorflow/python/util:nest",
#         "@absl_py//absl/flags",
#         "@absl_py//absl/testing:parameterized",
#     ],
# )
#
# py_strict_test(
#     name = "extensions_test_tpu",
#     srcs = ["extensions_test.py"],
#     args = [
#         "--jax_allow_unused_tpus",
#         "--requires_tpu",
#     ],
#     main = "extensions_test.py",
#     python_version = "PY3",
#     tags = [
#         "no_pip",
#         "requires-tpu",
#     ],
#     deps = [
#         ":extensions",
#         ":np_wrapper",
#         "//learning/brain/google/xla",
#         # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
#         "//third_party/py/jax",
#         "//third_party/py/numpy",
#         "//tensorflow:tensorflow_py",
#         "//tensorflow/python/distribute/cluster_resolver/tpu:tpu_cluster_resolver_py",
#         "//tensorflow/python/eager:backprop",
#         "//tensorflow/python/eager:context",
#         "//tensorflow/python/eager/polymorphic_function",
#         "//tensorflow/python/framework:config",
#         "//tensorflow/python/framework:constant_op",
#         "//tensorflow/python/framework:dtypes",
#         "//tensorflow/python/framework:ops",
#         "//tensorflow/python/framework:tensor",
#         "//tensorflow/python/framework:tensor_shape",
#         "//tensorflow/python/ops:array_ops",
#         "//tensorflow/python/ops:gradient_checker_v2",
#         "//tensorflow/python/ops:math_ops",
#         "//tensorflow/python/ops:nn_ops",
#         "//tensorflow/python/ops:random_ops",
#         "//tensorflow/python/ops:stateful_random_ops",
#         "//tensorflow/python/ops/numpy_ops:numpy",
#         "//tensorflow/python/platform:client_testlib",
#         "//tensorflow/python/util:nest",
#         "@absl_py//absl/flags",
#         "@absl_py//absl/testing:parameterized",
#     ],
# )
# copybara:uncomment_end

py_strict_test(
    name = "np_test",
    timeout = "long",
    srcs = ["np_test.py"],
    args = [
        "--num_generated_cases=90",
        "--enable_x64",  # Needed to enable dtype check
    ],
    python_version = "PY3",
    shard_count = 20,
    srcs_version = "PY2AND3",
    tags = [
        "gpu",
        "no_pip",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":config",
        ":extensions",
        ":np_wrapper",
        ":test_util",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/ops/numpy_ops:np_config",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "np_indexing_test",
    srcs = ["np_indexing_test.py"],
    args = [
        "--num_generated_cases=90",
        "--enable_x64",  # Needed to enable dtype check
    ],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY2AND3",
    # TODO(b/164245103): Re-enable GPU once tf.tensor_strided_slice_update's segfault is fixed.
    tags = [
        "no_pip",
        #     "gpu",
        #     "requires-gpu-nvidia",
    ],
    deps = [
        ":config",
        ":extensions",
        ":np_wrapper",
        ":test_util",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "//tensorflow/python/util:nest",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)

py_strict_test(
    name = "np_einsum_test",
    srcs = ["np_einsum_test.py"],
    args = [
        "--num_generated_cases=90",
        "--enable_x64",  # Needed to enable dtype check
    ],
    python_version = "PY3",
    shard_count = 20,
    srcs_version = "PY2AND3",
    tags = [
        "gpu",
        "no_pip",
        "requires-gpu-nvidia",
    ],
    deps = [
        ":config",
        ":np_wrapper",
        ":test_util",
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//third_party/py/numpy",
        "//tensorflow/python/ops/numpy_ops:np_config",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
