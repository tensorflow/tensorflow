load("//tensorflow/core/platform:rules_cc.bzl", "cc_library")
load(
    "//tensorflow:tensorflow.bzl",
    "if_cuda_or_rocm",
)
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")
load(
    "@local_config_rocm//rocm:build_defs.bzl",
    "if_rocm",
)

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "filegroup")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_cuda_cc_test")

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "tf_kernel_library")

package(
    default_visibility = [
        "//tensorflow:__subpackages__",
        "//tensorflow:internal",
    ],
    features = ["-layering_check"],
    licenses = ["notice"],
)

# TODO(rmlarsen): Remove ASAP.
package_group(
    name = "friends",
    packages = ["//tensorflow/..."],
)

# Export a few files for use on Android.
exports_files([
    "cholesky_op.cc",
    "determinant_op.cc",
    "einsum_op_impl_half.cc",
    "einsum_op_impl_bfloat16.cc",
    "einsum_op_impl_int32.cc",
    "einsum_op_impl_int64.cc",
    "einsum_op_impl_float.cc",
    "einsum_op_impl_double.cc",
    "einsum_op_impl_complex64.cc",
    "einsum_op_impl_complex128.cc",
    "einsum_op_impl.h",
    "einsum_op.h",
    "linalg_ops_common.h",
    "linalg_ops_common.cc",
    "matrix_band_part_op.h",
    "matrix_band_part_op.cc",
    "matrix_diag_op.h",
    "matrix_diag_op.cc",
    "matrix_inverse_op.cc",
    "matrix_set_diag_op.h",
    "matrix_set_diag_op.cc",
    "matrix_triangular_solve_op_complex.cc",
    "matrix_triangular_solve_op_impl.h",
    "matrix_triangular_solve_op_real.cc",
    "qr_op_complex128.cc",
    "qr_op_complex64.cc",
    "qr_op_double.cc",
    "qr_op_float.cc",
    "qr_op_impl.h",
])

# Public support libraries ----------------------------------------------------

cc_library(
    name = "linalg",
    deps = [
        ":banded_triangular_solve_op",
        ":cholesky_grad",
        ":cholesky_op",
        ":determinant_op",
        ":eig_op",
        ":einsum_op",
        ":lu_op",
        ":matrix_band_part_op",
        ":matrix_diag_op",
        ":matrix_exponential_op",
        ":matrix_inverse_op",
        ":matrix_logarithm_op",
        ":matrix_set_diag_op",
        ":matrix_solve_ls_op",
        ":matrix_solve_op",
        ":matrix_square_root_op",
        ":matrix_triangular_solve_op",
        ":qr_op",
        ":self_adjoint_eig_op",
        ":self_adjoint_eig_v2_op",
        ":svd_op",
        ":tridiagonal_matmul_op",
        ":tridiagonal_solve_op",
    ],
)

LINALG_DEPS = [
    ":linalg_ops_common",
    "//third_party/eigen3",
    "//tensorflow/core:framework",
    "//tensorflow/core:lib",
    "//tensorflow/core/kernels:cast_op",
    "//tensorflow/core/kernels:fill_functor",
    "//tensorflow/core/kernels:transpose_functor",
] + if_cuda_or_rocm([
    ":eye_functor",
]) + if_cuda([
    "//tensorflow/core/util:cuda_solvers",
]) + if_rocm([
    "//tensorflow/core/util:rocm_solvers",
])

tf_kernel_library(
    name = "matrix_band_part_op",
    prefix = "matrix_band_part_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "matrix_diag_op",
    prefix = "matrix_diag_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "matrix_set_diag_op",
    prefix = "matrix_set_diag_op",
    deps = LINALG_DEPS + [":matrix_diag_op"],
)

tf_kernel_library(
    name = "cholesky_op",
    prefix = "cholesky_op",
    deps = if_cuda_or_rocm([
        ":matrix_band_part_op",
    ]) + LINALG_DEPS,
)

tf_kernel_library(
    name = "cholesky_grad",
    prefix = "cholesky_grad",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "determinant_op",
    prefix = "determinant_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "matrix_exponential_op",
    prefix = "matrix_exponential_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "matrix_logarithm_op",
    prefix = "matrix_logarithm_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "self_adjoint_eig_op",
    prefix = "self_adjoint_eig_op",
    deps = LINALG_DEPS + ["//tensorflow/core:lib_internal"],
)

tf_kernel_library(
    name = "self_adjoint_eig_v2_op",
    prefix = "self_adjoint_eig_v2_op",
    deps = LINALG_DEPS + ["//tensorflow/core:lib_internal"] + if_cuda([
        "//tensorflow/core/kernels:cwise_op",
    ]),
)

tf_kernel_library(
    name = "eig_op",
    prefix = "eig_op",
    deps = LINALG_DEPS + ["//tensorflow/core:lib_internal"] + if_cuda([
        "//tensorflow/core/kernels:cwise_op",
    ]),
)

tf_kernel_library(
    name = "matrix_inverse_op",
    prefix = "matrix_inverse_op",
    visibility = [":friends"],
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "matrix_solve_ls_op",
    prefix = "matrix_solve_ls_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "matrix_solve_op",
    prefix = "matrix_solve_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "matrix_square_root_op",
    prefix = "matrix_square_root_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "banded_triangular_solve_op",
    prefix = "banded_triangular_solve_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "matrix_triangular_solve_op",
    hdrs = ["matrix_triangular_solve_op_impl.h"],
    prefix = "matrix_triangular_solve_op",
    deps = [
        ":linalg_ops_common",
        "//third_party/eigen3",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:fill_functor",
        "//tensorflow/core/platform:stream_executor",
    ] + if_cuda([
        "//tensorflow/core/platform/default/build_config:cublas_plugin",
        "//tensorflow/core/util:cuda_solvers",
    ]) + if_rocm([
        "@local_config_rocm//rocm:rocprim",
        "//tensorflow/core/util:rocm_solvers",
    ]) + if_cuda_or_rocm([
        "//tensorflow/core/kernels:transpose_functor",
    ]),
)

tf_kernel_library(
    name = "tridiagonal_matmul_op",
    srcs = ["tridiagonal_matmul_op.cc"],
    gpu_srcs = ["tridiagonal_matmul_op_gpu.cu.cc"],
    deps = LINALG_DEPS + if_cuda([
        "//tensorflow/core/util:cuda_sparse",
    ]),
)

tf_kernel_library(
    name = "tridiagonal_solve_op",
    srcs = ["tridiagonal_solve_op.cc"],
    gpu_srcs = ["tridiagonal_solve_op_gpu.cu.cc"],
    deps = LINALG_DEPS + if_cuda([
        "//tensorflow/core/util:cuda_sparse",
    ]),
)

tf_kernel_library(
    name = "qr_op",
    prefix = "qr_op",
    deps = LINALG_DEPS + if_cuda_or_rocm([
        "//tensorflow/core/kernels:cwise_op",
        ":matrix_band_part_op",
    ]),
)

tf_kernel_library(
    name = "svd_op",
    prefix = "svd_op",
    deps = LINALG_DEPS,
)

tf_kernel_library(
    name = "lu_op",
    prefix = "lu_op",
    deps = if_cuda_or_rocm([
        "//tensorflow/core/kernels:transpose_functor",
    ]) + if_cuda([
        "//tensorflow/core/util:cuda_solvers",
    ]) + if_rocm([
        "//tensorflow/core/util:rocm_solvers",
    ]) + [
        "//third_party/eigen3",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

tf_kernel_library(
    name = "einsum_op",
    prefix = "einsum_op",
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core/kernels:batch_matmul_op",
        "//tensorflow/core/kernels:fill_functor",
        "//tensorflow/core/kernels:reduction_ops",
        "//tensorflow/core/kernels:transpose_functor",
        "//tensorflow/core/profiler/lib:traceme",
        "//third_party/eigen3",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "linalg_ops_common",
    srcs = ["linalg_ops_common.cc"],
    hdrs = ["linalg_ops_common.h"],
    visibility = ["//visibility:private"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//third_party/eigen3",
    ],
)

tf_cuda_cc_test(
    name = "banded_triangular_solve_op_test",
    size = "small",
    srcs = ["banded_triangular_solve_op_test.cc"],
    deps = [
        ":banded_triangular_solve_op",
        ":matrix_set_diag_op",
        ":matrix_triangular_solve_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

tf_kernel_library(
    name = "eye_functor",
    hdrs = ["eye_functor.h"],
    gpu_srcs = [
        "eye_functor_gpu.cu.cc",
        "eye_functor.h",
    ],
    visibility = ["//tensorflow/core/kernels:friends"],
    deps = [
        "//tensorflow/core:framework",
        "//third_party/eigen3",
    ],
    alwayslink = 0,
)

tf_cuda_cc_test(
    name = "matrix_triangular_solve_op_test",
    size = "small",
    srcs = ["matrix_triangular_solve_op_test.cc"],
    deps = [
        ":matrix_triangular_solve_op",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
        "//tensorflow/core/kernels:broadcast_to_op",
        "//tensorflow/core/kernels:ops_testutil",
        "//tensorflow/core/kernels:ops_util",
    ],
)

# A file group which contains all operators which are known to work on mobile.
filegroup(
    name = "portable_all_op_kernels",
    srcs = glob(
        [
            "*.cc",
            "*.h",
        ],
        exclude = [
            "*test.cc",
            "*test.h",
            "*_test_*",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
