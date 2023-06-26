load("//tensorflow:strict.default.bzl", "py_strict_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "linalg_impl",
    srcs = ["linalg_impl.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:constant_op",
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_shape",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:cond",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:linalg_ops",
        "//tensorflow/python/ops:linalg_ops_gen",
        "//tensorflow/python/ops:map_fn",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:special_math_ops",
        "//tensorflow/python/ops:stateless_random_ops",
        "//tensorflow/python/ops:while_loop",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_util",
    srcs = ["linear_operator_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow/python/framework:dtypes",
        "//tensorflow/python/framework:ops",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:linalg_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_low_rank_update",
    srcs = ["linear_operator_low_rank_update.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_diag",
        ":linear_operator_identity",
        ":linear_operator_util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:linalg_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator_circulant",
    srcs = ["linear_operator_circulant.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops/distributions:util",
        "//tensorflow/python/ops/signal:fft_ops",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_identity",
    srcs = ["linear_operator_identity.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_lower_triangular",
    srcs = ["linear_operator_lower_triangular.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator_kronecker",
    srcs = ["linear_operator_kronecker.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        "//tensorflow/python/framework:common_shapes",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "registrations_util",
    srcs = ["registrations_util.py"],
)

py_strict_library(
    name = "linear_operator_block_lower_triangular",
    srcs = ["linear_operator_block_lower_triangular.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_algebra",
        ":linear_operator_util",
        "//tensorflow/python/framework:common_shapes",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "matmul_registrations",
    srcs = ["matmul_registrations.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_algebra",
        ":linear_operator_block_diag",
        ":linear_operator_circulant",
        ":linear_operator_composition",
        ":linear_operator_diag",
        ":linear_operator_identity",
        ":linear_operator_lower_triangular",
        ":linear_operator_zeros",
        ":registrations_util",
    ],
)

py_strict_library(
    name = "linear_operator",
    srcs = ["linear_operator.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator_algebra",
        ":linear_operator_util",
        ":slicing",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:composite_tensor_gradient",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/framework:tensor_spec",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:type_spec",
        "//tensorflow/python/framework:type_spec_registry",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:linalg_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:resource_variable_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/python/trackable:data_structures",
        "//tensorflow/python/util:deprecation",
        "//tensorflow/python/util:dispatch",
        "//tensorflow/python/util:nest",
        "//tensorflow/python/util:tf_export",
        "//tensorflow/python/util:variable_utils",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_inversion",
    srcs = ["linear_operator_inversion.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator_algebra",
    srcs = ["linear_operator_algebra.py"],
    deps = [
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/util:tf_inspect",
    ],
)

py_strict_library(
    name = "adjoint_registrations",
    srcs = ["adjoint_registrations.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_adjoint",
        ":linear_operator_algebra",
        ":linear_operator_block_diag",
        ":linear_operator_circulant",
        ":linear_operator_diag",
        ":linear_operator_householder",
        ":linear_operator_identity",
        ":linear_operator_kronecker",
        "//tensorflow/python/ops:math_ops",
    ],
)

py_strict_library(
    name = "linear_operator_tridiag",
    srcs = ["linear_operator_tridiag.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_gen",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:manip_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator_toeplitz",
    srcs = ["linear_operator_toeplitz.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_circulant",
        ":linear_operator_util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops/signal:fft_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator_block_diag",
    srcs = ["linear_operator_block_diag.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_algebra",
        ":linear_operator_util",
        "//tensorflow/python/framework:common_shapes",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "cholesky_registrations",
    srcs = ["cholesky_registrations.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_algebra",
        ":linear_operator_block_diag",
        ":linear_operator_composition",
        ":linear_operator_diag",
        ":linear_operator_identity",
        ":linear_operator_kronecker",
        ":linear_operator_lower_triangular",
        ":linear_operator_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:linalg_ops",
        "//tensorflow/python/ops:math_ops",
    ],
)

py_strict_library(
    name = "linear_operator_test_util",
    srcs = ["linear_operator_test_util.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator_util",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/eager:def_function",
        "//tensorflow/python/framework:composite_tensor",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:random_seed",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/framework:test_lib",
        "//tensorflow/python/module",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:gradients_impl",
        "//tensorflow/python/ops:linalg_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:random_ops",
        "//tensorflow/python/ops:sort_ops",
        "//tensorflow/python/ops:variables",
        "//tensorflow/python/ops:while_v2",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/saved_model:load",
        "//tensorflow/python/saved_model:nested_structure_coder",
        "//tensorflow/python/saved_model:save",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_permutation",
    srcs = ["linear_operator_permutation.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:sort_ops",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "inverse_registrations",
    srcs = ["inverse_registrations.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_addition",
        ":linear_operator_algebra",
        ":linear_operator_block_diag",
        ":linear_operator_block_lower_triangular",
        ":linear_operator_circulant",
        ":linear_operator_diag",
        ":linear_operator_full_matrix",
        ":linear_operator_householder",
        ":linear_operator_identity",
        ":linear_operator_inversion",
        ":linear_operator_kronecker",
        "//tensorflow/python/ops:math_ops",
    ],
)

py_strict_library(
    name = "linalg",
    srcs = ["linalg.py"],
    deps = [
        ":adjoint_registrations",
        ":cholesky_registrations",
        ":inverse_registrations",
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_algebra",
        ":linear_operator_block_diag",
        ":linear_operator_block_lower_triangular",
        ":linear_operator_circulant",
        ":linear_operator_composition",
        ":linear_operator_diag",
        ":linear_operator_full_matrix",
        ":linear_operator_identity",
        ":linear_operator_kronecker",
        ":linear_operator_low_rank_update",
        ":linear_operator_lower_triangular",
        ":linear_operator_permutation",
        ":linear_operator_toeplitz",
        ":linear_operator_tridiag",
        ":linear_operator_zeros",
        ":matmul_registrations",
        ":solve_registrations",
    ],
)

py_strict_library(
    name = "linear_operator_full_matrix",
    srcs = ["linear_operator_full_matrix.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator_householder",
    srcs = ["linear_operator_householder.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/ops:nn",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator_adjoint",
    srcs = ["linear_operator_adjoint.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "solve_registrations",
    srcs = ["solve_registrations.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_algebra",
        ":linear_operator_block_diag",
        ":linear_operator_circulant",
        ":linear_operator_composition",
        ":linear_operator_diag",
        ":linear_operator_identity",
        ":linear_operator_inversion",
        ":linear_operator_lower_triangular",
        ":registrations_util",
    ],
)

py_strict_library(
    name = "linear_operator_composition",
    srcs = ["linear_operator_composition.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_util",
        "//tensorflow/python/framework:common_shapes",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "slicing",
    srcs = ["slicing.py"],
    deps = [
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/util:nest",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_zeros",
    srcs = ["linear_operator_zeros.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        "//tensorflow/python/framework:errors",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_util",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:array_ops_stack",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:control_flow_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:tf_export",
        "//third_party/py/numpy",
    ],
)

py_strict_library(
    name = "linear_operator_diag",
    srcs = ["linear_operator_diag.py"],
    deps = [
        ":linalg_impl",
        ":linear_operator",
        ":linear_operator_util",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/framework:tensor_conversion",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
        "//tensorflow/python/ops:math_ops",
        "//tensorflow/python/util:tf_export",
    ],
)

py_strict_library(
    name = "linear_operator_addition",
    srcs = ["linear_operator_addition.py"],
    deps = [
        ":linear_operator",
        ":linear_operator_diag",
        ":linear_operator_full_matrix",
        ":linear_operator_identity",
        ":linear_operator_lower_triangular",
        "//tensorflow/python/framework:for_generated_wrappers",
        "//tensorflow/python/ops:array_ops",
        "//tensorflow/python/ops:check_ops",
    ],
)
