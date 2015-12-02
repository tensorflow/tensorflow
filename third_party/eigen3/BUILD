licenses(["restricted"])  # MPL2, portions GPL v3, LGPL v3, BSD-like

cc_library(
    name = "eigen3",
    hdrs = glob([
        "**/*.h",
        "unsupported/Eigen/CXX11/*",
        "Eigen/*",
    ]),
    includes = [ "." ],
    visibility = ["//visibility:public"],
)
