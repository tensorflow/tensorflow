# Description:
#   Eigen is a C++ template library for linear algebra: vectors,
#   matrices, and related algorithms.
# This is the BUILD file used for the @eigen_archive external repository.

licenses([
    # Note: Although Eigen also includes GPL V3 and LGPL v2.1+ code, TensorFlow
    #       has taken special care to not reference any restricted code.
    "reciprocal",  # MPL2
    "notice",  # Portions BSD
])

exports_files(["COPYING.MPL2"])

EIGEN_FILES = [
    "Eigen/**",
    "unsupported/Eigen/CXX11/**",
    "unsupported/Eigen/FFT",
    "unsupported/Eigen/KroneckerProduct",
    "unsupported/Eigen/src/FFT/**",
    "unsupported/Eigen/src/KroneckerProduct/**",
    "unsupported/Eigen/MatrixFunctions",
    "unsupported/Eigen/SpecialFunctions",
    "unsupported/Eigen/src/MatrixFunctions/**",
    "unsupported/Eigen/src/SpecialFunctions/**",
]

# Files known to be under MPL2 license.
EIGEN_MPL2_HEADER_FILES = glob(
    EIGEN_FILES,
    exclude = [
        # Guarantees that any non-MPL2 file added to the list above will fail to
        # compile.
        "Eigen/src/Core/util/NonMPL2.h",
        "Eigen/**/CMakeLists.txt",
    ],
)

cc_library(
    name = "eigen",
    hdrs = EIGEN_MPL2_HEADER_FILES,
    defines = [
        # This define (mostly) guarantees we don't link any problematic
        # code. We use it, but we do not rely on it, as evidenced above.
        "EIGEN_MPL2_ONLY",
        "EIGEN_MAX_ALIGN_BYTES=64",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "eigen_header_files",
    srcs = EIGEN_MPL2_HEADER_FILES,
    visibility = ["//visibility:public"],
)
