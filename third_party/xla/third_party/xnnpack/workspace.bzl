"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "00fdf2a6331161484b1cdab26c022d14a7739765d8c905ab9d6da6a1b2cace46",
        strip_prefix = "XNNPACK-613e531bf8199f490af7e2a534140c721e434a90",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/613e531bf8199f490af7e2a534140c721e434a90.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
