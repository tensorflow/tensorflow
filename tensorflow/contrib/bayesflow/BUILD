# Description:
#   Contains ops for working with statistical distributions,
#   particularly useful for Bayesian inference.
#   APIs here are meant to evolve over time.

package(default_visibility = [
    "//learning/brain/contrib/bayesflow:__subpackages__",
    "//tensorflow:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "bayesflow_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:functional_ops",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:util",
        "//third_party/py/numpy",
    ],
)

cuda_py_test(
    name = "metropolis_hastings_test",
    size = "large",
    srcs = ["python/kernel_tests/metropolis_hastings_test.py"],
    additional_deps = [
        ":bayesflow_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "csiszar_divergence_test",
    size = "medium",
    srcs = ["python/kernel_tests/csiszar_divergence_test.py"],
    additional_deps = [
        ":bayesflow_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
    ],
    tags = [
        "manual",  # b/64490288
        "notap",
    ],
)

cuda_py_test(
    name = "custom_grad_test",
    size = "small",
    srcs = ["python/kernel_tests/custom_grad_test.py"],
    additional_deps = [
        ":bayesflow_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "docstring_util_test",
    size = "small",
    srcs = ["python/kernel_tests/docstring_util_test.py"],
    additional_deps = [
        ":bayesflow_py",
        "//tensorflow/python:client_testlib",
    ],
)

cuda_py_test(
    name = "layers_conv_variational_test",
    size = "small",
    srcs = ["python/kernel_tests/layers_conv_variational_test.py"],
    additional_deps = [
        ":bayesflow_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "layers_dense_variational_test",
    size = "small",
    srcs = ["python/kernel_tests/layers_dense_variational_test.py"],
    additional_deps = [
        ":bayesflow_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
    ],
)

cuda_py_test(
    name = "mcmc_diagnostics_test",
    size = "small",
    srcs = ["python/kernel_tests/mcmc_diagnostics_test.py"],
    additional_deps = [
        ":bayesflow_py",
        "//third_party/py/numpy",
        "//tensorflow/python:spectral_ops_test_util",
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_seed",
    ],
)

cuda_py_test(
    name = "monte_carlo_test",
    size = "small",
    srcs = ["python/kernel_tests/monte_carlo_test.py"],
    additional_deps = [
        ":bayesflow_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_seed",
    ],
)

cuda_py_test(
    name = "halton_sequence_test",
    size = "small",
    srcs = ["python/kernel_tests/halton_sequence_test.py"],
    additional_deps = [
        ":bayesflow_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
    ],
    tags = ["no_mac"],  # b/73192243
)

cuda_py_test(
    name = "hmc_test",
    size = "large",
    srcs = ["python/kernel_tests/hmc_test.py"],
    additional_deps = [
        ":bayesflow_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_seed",
    ],
)

cuda_py_test(
    name = "sgld_optimizer_test",
    size = "small",
    srcs = ["python/kernel_tests/sgld_optimizer_test.py"],
    additional_deps = [
        ":bayesflow_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_seed",
    ],
    tags = ["notsan"],
)

cuda_py_test(
    name = "variational_sgd_optimizer_test",
    size = "small",
    srcs = ["python/kernel_tests/variational_sgd_optimizer_test.py"],
    additional_deps = [
        ":bayesflow_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/distributions:distributions_py",
        "//tensorflow/contrib/layers:layers_py",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_seed",
    ],
    tags = ["notsan"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
