licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

cc_library(
    name = "eigen3",
    hdrs = glob([
        "Eigen/Core",
        "Eigen/LU",
        "Eigen/Cholesky",
        "Eigen/Eigenvalues",
        "Eigen/QR",
        "unsupported/Eigen/CXX11/Tensor",
        "unsupported/Eigen/CXX11/FixedPoint",
        "unsupported/Eigen/CXX11/src/FixedPoint/*.h",
    ]),
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen_archive//:eigen",
    ],
)
