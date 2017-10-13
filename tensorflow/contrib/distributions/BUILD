# Description:
#   Contains ops for statistical distributions (with pdf, cdf, sample, etc...).
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "cuda_py_test")

py_library(
    name = "bijectors_py",
    srcs = glob(["python/ops/bijectors/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "distributions_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
    deps = [
        ":bijectors_py",
        "//tensorflow/contrib/framework:framework_py",
        "//tensorflow/contrib/learn",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:data_flow_ops",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:histogram_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:linalg_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:state_ops",
        "//tensorflow/python:tensor_util",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python/ops/distributions",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

cuda_py_test(
    name = "estimator_test",
    size = "small",
    srcs = ["python/kernel_tests/estimator_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/learn",
        "//tensorflow/contrib/learn:head_test",
        "//tensorflow/python/ops/distributions",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:session",
    ],
    tags = ["no_pip"],  # contrib/learn:head_test is not available in pip.
)

cuda_py_test(
    name = "distribution_test",
    size = "small",
    srcs = ["python/kernel_tests/distribution_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
    ],
)

cuda_py_test(
    name = "conditional_distribution_test",
    size = "small",
    srcs = [
        "python/kernel_tests/conditional_distribution_test.py",
        "python/kernel_tests/distribution_test.py",
    ],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
    ],
    tags = ["no_pip"],
)

cuda_py_test(
    name = "binomial_test",
    size = "small",
    srcs = ["python/kernel_tests/binomial_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "chi2_test",
    srcs = ["python/kernel_tests/chi2_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "deterministic_test",
    size = "small",
    srcs = ["python/kernel_tests/deterministic_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "geometric_test",
    size = "small",
    srcs = ["python/kernel_tests/geometric_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "inverse_gamma_test",
    srcs = ["python/kernel_tests/inverse_gamma_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "moving_stats_test",
    size = "small",
    srcs = ["python/kernel_tests/moving_stats_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
    tags = ["nomsan"],  # disable to avoid false positives from scipy.
)

cuda_py_test(
    name = "mvn_diag_test",
    size = "small",
    srcs = ["python/kernel_tests/mvn_diag_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "mvn_diag_plus_low_rank_test",
    size = "medium",
    srcs = ["python/kernel_tests/mvn_diag_plus_low_rank_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "mvn_full_covariance_test",
    size = "small",
    srcs = ["python/kernel_tests/mvn_full_covariance_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "mvn_tril_test",
    size = "small",
    srcs = ["python/kernel_tests/mvn_tril_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "mixture_test",
    size = "medium",
    srcs = ["python/kernel_tests/mixture_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:random_ops",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "mixture_same_family_test",
    size = "small",
    srcs = ["python/kernel_tests/mixture_same_family_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:client_testlib",
    ],
)

cuda_py_test(
    name = "negative_binomial_test",
    size = "small",
    srcs = ["python/kernel_tests/negative_binomial_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "poisson_test",
    size = "small",
    srcs = ["python/kernel_tests/poisson_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "poisson_lognormal_test",
    size = "small",
    srcs = ["python/kernel_tests/poisson_lognormal_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "sinh_arcsinh_test",
    size = "small",
    srcs = ["python/kernel_tests/sinh_arcsinh_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "independent_test",
    size = "small",
    srcs = ["python/kernel_tests/independent_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "sample_stats_test",
    size = "medium",
    srcs = ["python/kernel_tests/sample_stats_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:platform_test",
    ],
    tags = ["nomsan"],  # disable to avoid false positives from scipy.
)

cuda_py_test(
    name = "vector_sinh_arcsinh_diag_test",
    size = "medium",
    srcs = ["python/kernel_tests/vector_sinh_arcsinh_diag_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "vector_exponential_diag_test",
    size = "medium",
    srcs = ["python/kernel_tests/vector_exponential_diag_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "vector_laplace_diag_test",
    size = "medium",
    srcs = ["python/kernel_tests/vector_laplace_diag_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "vector_student_t_test",
    size = "medium",
    srcs = ["python/kernel_tests/vector_student_t_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "wishart_test",
    size = "small",
    srcs = ["python/kernel_tests/wishart_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "normal_conjugate_posteriors_test",
    size = "small",
    srcs = ["python/kernel_tests/normal_conjugate_posteriors_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "quantized_distribution_test",
    size = "medium",
    srcs = ["python/kernel_tests/quantized_distribution_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
        "//tensorflow/python:variables",
    ],
)

cuda_py_test(
    name = "logistic_test",
    size = "small",
    srcs = ["python/kernel_tests/logistic_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "relaxed_bernoulli_test",
    size = "small",
    srcs = ["python/kernel_tests/relaxed_bernoulli_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "onehot_categorical_test",
    size = "small",
    srcs = ["python/kernel_tests/onehot_categorical_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "relaxed_onehot_categorical_test",
    size = "small",
    srcs = ["python/kernel_tests/relaxed_onehot_categorical_test.py"],
    additional_deps = [
        ":distributions_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "transformed_distribution_test",
    size = "medium",
    srcs = ["python/kernel_tests/transformed_distribution_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "vector_diffeomixture_test",
    size = "medium",
    srcs = ["python/kernel_tests/vector_diffeomixture_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:client_testlib",
    ],
)

cuda_py_test(
    name = "conditional_transformed_distribution_test",
    size = "medium",
    srcs = [
        "python/kernel_tests/conditional_transformed_distribution_test.py",
        "python/kernel_tests/transformed_distribution_test.py",
    ],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    tags = ["no_pip"],
)

cuda_py_test(
    name = "distribution_util_test",
    size = "small",
    srcs = ["python/kernel_tests/distribution_util_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:gradients",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "shape_test",
    size = "medium",
    srcs = ["python/kernel_tests/shape_test.py"],
    additional_deps = [
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
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

# === Bijector Tests ==========================================================

cuda_py_test(
    name = "conditional_bijector_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/conditional_bijector_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "absolute_value_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/absolute_value_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "affine_test",
    size = "large",
    srcs = ["python/kernel_tests/bijectors/affine_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
    shard_count = 5,
    tags = ["noasan"],  # times out b/63678675
)

cuda_py_test(
    name = "affine_linear_operator_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/affine_linear_operator_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "chain_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/chain_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "cholesky_outer_product_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/cholesky_outer_product_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "exp_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/exp_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "inline_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/inline_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "invert_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/invert_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "permute_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/permute_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "power_transform_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/power_transform_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "sigmoid_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/sigmoid_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "sigmoid_centered_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/sigmoid_centered_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

# Tests for SinhArcSinh bijector.  The file name has the extra "_bijector" to
# avoid BUILD rule name conflicts with the distribution by the same name.
cuda_py_test(
    name = "sinh_arcsinh_bijector_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/sinh_arcsinh_bijector_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "softmax_centered_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/softmax_centered_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "softplus_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/softplus_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)

cuda_py_test(
    name = "weibull_test",
    size = "small",
    srcs = ["python/kernel_tests/bijectors/weibull_test.py"],
    additional_deps = [
        ":bijectors_py",
        ":distributions_py",
        "//third_party/py/numpy",
        "@six_archive//:six",
        "//tensorflow/contrib/linalg:linalg_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_for_generated_wrappers",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:platform_test",
    ],
)
