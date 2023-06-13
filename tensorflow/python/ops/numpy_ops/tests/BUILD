# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

licenses(["notice"])

py_library(
    name = "config",
    srcs = ["config.py"],
    srcs_version = "PY2AND3",
    deps = [
    ],
)

py_library(
    name = "test_util",
    srcs = ["test_util.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":config",
        ":extensions",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "np_wrapper",
    srcs = ["np_wrapper.py"],
    srcs_version = "PY2AND3",
    visibility = [
        "//visibility:public",
    ],
    deps = [
        "//tensorflow/python/compat:v2_compat",
        "//tensorflow/python/ops/numpy_ops:np_config",
        "//tensorflow/python/ops/numpy_ops:np_dtypes",
        "//tensorflow/python/ops/numpy_ops:numpy",
    ],
)

py_library(
    name = "extensions",
    srcs = ["extensions.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":np_wrapper",
        "//tensorflow:tensorflow_py",
        "@six_archive//:six",
    ],
)

# copybara:uncomment_begin(google-only)
# py_test(
#     name = "extensions_test",
#     srcs = ["extensions_test.py"],
#     python_version = "PY3",
#     srcs_version = "PY2AND3",
#     tags = [
#         "gpu",
#         "no_pip",
#         "requires-gpu-nvidia",
#     ],
#     deps = [
#         ":extensions",
#         ":np_wrapper",
#         "//learning/brain/research/jax:gpu_support",
#         "//third_party/py/jax",
#         "//tensorflow:tensorflow_py",
#     ],
# )
#
# py_test(
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
#         "//third_party/py/jax",
#         "//tensorflow:tensorflow_py",
#         "@absl_py//absl/flags",
#     ],
# )
# copybara:uncomment_end

py_test(
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
        ":np_wrapper",
        ":test_util",
    ],
)

py_test(
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
        ":np_wrapper",
        ":test_util",
    ],
)

py_test(
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
        ":test_util",
        "//tensorflow/python/ops/numpy_ops:np_config",
        "//tensorflow/python/ops/numpy_ops:numpy",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
    ],
)
