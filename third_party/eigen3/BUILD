# Description:
#   Eigen is a C++ template library for linear algebra: vectors,
#   matrices, and related algorithms.

licenses([
    # Note: Eigen is an MPL2 library that includes GPL v3 and LGPL v2.1+ code.
    #       We've taken special care to not reference any restricted code.
    "reciprocal",  # MPL2
    "notice",  # Portions BSD
])

exports_files(["LICENSE"])

# INTEL_MKL start
load("//tensorflow:tensorflow.bzl", "if_mkl")

# INTEL_MKL end
load("//tensorflow:tensorflow.bzl", "if_mkl")

EIGEN3_THIRD_PARTY_HEADERS = [
    "Eigen/Core",
    "Eigen/LU",
    "Eigen/Cholesky",
    "Eigen/Eigenvalues",
    "Eigen/QR",
    "Eigen/SVD",
    "unsupported/Eigen/MatrixFunctions",
    "unsupported/Eigen/SpecialFunctions",
    "unsupported/Eigen/CXX11/ThreadPool",
    "unsupported/Eigen/CXX11/Tensor",
    "unsupported/Eigen/CXX11/FixedPoint",
] + glob(["unsupported/Eigen/CXX11/src/FixedPoint/*.h"])

cc_library(
    name = "eigen3",
    hdrs = EIGEN3_THIRD_PARTY_HEADERS,
    includes = if_mkl(["./mkl_include"]),
    visibility = ["//visibility:public"],
    deps = [
        "@eigen_archive//:eigen",
        "@local_config_sycl//sycl",
    ],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = ["**/OWNERS"],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "eigen_third_party_header_files",
    srcs = EIGEN3_THIRD_PARTY_HEADERS,
    visibility = ["//visibility:public"],
)

genrule(
    name = "install_eigen_headers",
    srcs = [
        "@eigen_archive//:eigen_header_files",
        ":eigen_third_party_header_files",
    ],
    outs = ["include"],
    cmd = """
    mkdir $@
    for f in $(SRCS); do
      d="$${f%/*}"
      d="$${d#*external/eigen_archive/}"

      mkdir -p "$@/$${d}"
      cp "$${f}" "$@/$${d}/"
    done
    """,
    tags = ["manual"],
)
