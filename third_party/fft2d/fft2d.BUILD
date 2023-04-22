# 2D Fast Fourier Transform package
# from http://momonga.t.u-tokyo.ac.jp/~ooura/fft2d.html

package(
    default_visibility = ["//visibility:public"],
)

# Unrestricted use; can only distribute original package.
licenses(["notice"])

exports_files(["readme2d.txt"])

FFT2D_SRCS = [
    "fftsg.c",
    "fftsg2d.c",
]

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

# This is the main 2D FFT library.  The 2D FFTs in this library call
# 1D FFTs.  In addition, fast DCTs are provided for the special case
# of 8x8 and 16x16.  This code in this library is referred to as
# "Version II" on http://momonga.t.u-tokyo.ac.jp/~ooura/fft2d.html.
cc_library(
    name = "fft2d",
    srcs = FFT2D_SRCS,
    linkopts = select({
        ":windows": [],
        "//conditions:default": ["-lm"],
    }),
)

objc_library(
    name = "fft2d_ios",
    srcs = FFT2D_SRCS,
)

# Export the source code so that it could be compiled for Andoid native apps.
filegroup(
    name = "fft2d_srcs",
    srcs = FFT2D_SRCS,
)
