# Description:
#   Eigen is a C++ template library for linear algebra: vectors,
#   matrices, and related algorithms.
# This is the BUILD file used for the @eigen_archive external repository.

licenses([
    "reciprocal",  # MPL2
    "notice",  # Portions BSD
])

exports_files(["COPYING.MPL2"])

ALL_FILES_WITH_EXTENSIONS = glob(["**/*.*"])

# Top-level headers, excluding anything in one of the  ../src/.. directories.
EIGEN_HEADERS = glob(
    [
        "Eigen/*",
        "unsupported/Eigen/*",
        "unsupported/Eigen/CXX11/*",
    ],
    exclude = [
        "**/src/**",
    ] + ALL_FILES_WITH_EXTENSIONS,
)

# Internal eigen headers.
EIGEN_SOURCES = glob(
    [
        "Eigen/**/src/**/*.h",
        "Eigen/**/src/**/*.inc",
        "unsupported/Eigen/**/src/**/*.h",
        "unsupported/Eigen/**/src/**/*.inc",
    ],
)

cc_library(
    name = "eigen3",
    srcs = EIGEN_SOURCES,
    hdrs = EIGEN_HEADERS,
    defines = [
        "EIGEN_MAX_ALIGN_BYTES=64",
        "EIGEN_ALLOW_UNALIGNED_SCALARS",  # TODO(b/296071640): Remove when underlying bugs are fixed.
        "EIGEN_USE_AVX512_GEMM_KERNELS=0",  # TODO(b/238649163): Remove this once no longer necessary.
    ],
    includes = [
        ".",  # Third-party libraries include eigen relative to its root.
        "./mkl_include",  # For using MKL backend for Eigen when available.
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "eigen_header_files",
    srcs = EIGEN_HEADERS,
    visibility = ["//visibility:public"],
)

filegroup(
    name = "eigen_source_files",
    srcs = EIGEN_SOURCES,
    visibility = ["//visibility:public"],
)
