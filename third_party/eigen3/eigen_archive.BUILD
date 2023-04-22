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

# Internal eigen headers, known to be under an MPL2 license.
EIGEN_MPL2_SOURCES = glob(
    [
        "Eigen/**/src/**/*.h",
        "unsupported/Eigen/**/src/**/*.h",
    ],
    exclude = [
        # This guarantees that any file depending on non MPL2 licensed code
        # will not compile.
        "Eigen/src/Core/util/NonMPL2.h",
    ],
)

cc_library(
    name = "eigen3",
    srcs = EIGEN_MPL2_SOURCES,
    hdrs = EIGEN_HEADERS,
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
    srcs = EIGEN_HEADERS,
    visibility = ["//visibility:public"],
)

filegroup(
    name = "eigen_source_files",
    srcs = EIGEN_MPL2_SOURCES,
    visibility = ["//visibility:public"],
)
