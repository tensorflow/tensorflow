licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

cc_library(
    name = "eigen3",
    hdrs = glob([
        "Eigen/Core",
        "Eigen/LU",
        "Eigen/Cholesky",
        "unsupported/Eigen/CXX11/FixedPoint/Tensor",
        "unsupported/Eigen/CXX11/FixedPoint",
        "unsupported/Eigen/CXX11/src/FixedPoint/*.h",
        "unsupported/Eigen/CXX11/NeuralNetworks",
        "unsupported/Eigen/CXX11/src/NeuralNetworks/*.h",
    ]),
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        "@eigen_archive//:eigen",
    ],
)
