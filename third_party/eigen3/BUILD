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

cc_library(
    name = "eigen3",
    hdrs = glob(["unsupported/Eigen/CXX11/src/FixedPoint/*.h"]) + [
        "Eigen/Core",
        "Eigen/LU",
        "Eigen/Cholesky",
        "Eigen/Eigenvalues",
        "Eigen/QR",
        "Eigen/SVD",
        "unsupported/Eigen/SpecialFunctions",
        "unsupported/Eigen/CXX11/Tensor",
        "unsupported/Eigen/CXX11/FixedPoint",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen_archive//:eigen",
        "@local_config_sycl//sycl:sycl",
    ],
)
