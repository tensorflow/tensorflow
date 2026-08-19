load("@xla//xla/tsl:tsl.bzl", "if_macos")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

exports_files(["LICENSE"])

# Having fp-contract set to fast causes incorrect results on macos:
# https://github.com/jax-ml/jax/issues/28217
fp_contract = if_macos(
    ["-ffp-contract=on"],
    ["-ffp-contract=fast"],
)

DUCC_COPTS = [
    "-frtti",
    "-fexceptions",
] + fp_contract

# This library exposes the raw DUCC fft API.  It should be used
# with caution, since inclusion of the headers will require any
# dependent targets to be build with exceptions and RTTI enabled.
# For a better-isolated target, use ":fft_wrapper".
cc_library(
    name = "fft",
    srcs = [
        "google/ducc0_custom_lowlevel_threading.h",
        "google/threading.cc",
        "src/ducc0/infra/aligned_array.h",
        "src/ducc0/infra/error_handling.h",
        "src/ducc0/infra/misc_utils.h",
        "src/ducc0/infra/simd.h",
        "src/ducc0/infra/string_utils.h",
        "src/ducc0/infra/threading.cc",
        "src/ducc0/infra/useful_macros.h",
        "src/ducc0/math/cmplx.h",
        "src/ducc0/math/unity_roots.h",
    ],
    hdrs = [
        "google/threading.h",
        "src/ducc0/fft/fft.h",
        "src/ducc0/fft/fft1d_impl.h",
        "src/ducc0/fft/fftnd_impl.h",
        "src/ducc0/infra/mav.h",
        "src/ducc0/infra/threading.h",
    ],
    copts = DUCC_COPTS,
    defines = [
        # Use custom TSL/Eigen threading.
        "DUCC0_CUSTOM_LOWLEVEL_THREADING=1",
    ],
    features = ["-use_header_modules"],
    include_prefix = "ducc",
    includes = [
        ".",  # Needed for google/-relative paths.
        "google",  # Needed for finding ducc0_custom_lowlevel_threading.h.
        "src",  # Needed for internal headers.
    ],
    # The DUCC FFT source files are dual-licensed as BSD 3 clause and GPLv2.
    # We choose BSD 3 clause.
    licenses = ["notice"],
    visibility = ["//visibility:private"],
    deps = [
        # Required for custom threadpool usage:
        "@eigen_archive//:eigen3",
        "@tsl//tsl/platform:mutex",
    ],
)

cc_library(
    name = "fft_wrapper",
    srcs = ["google/fft.cc"],
    hdrs = ["google/fft.h"],
    copts = DUCC_COPTS,
    features = ["-use_header_modules"],
    include_prefix = "ducc",
    licenses = ["notice"],
    visibility = ["//visibility:public"],
    deps = [
        ":fft",
        "@eigen_archive//:eigen3",
    ],
)
