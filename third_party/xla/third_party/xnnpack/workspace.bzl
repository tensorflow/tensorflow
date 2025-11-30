"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "f644ad3ac88b3b0208a82742938bca35235865d6ca64950dac58b166877eb2a5",
        strip_prefix = "XNNPACK-1b918df9d1744ae40725254f4baa592ed05c912e",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/1b918df9d1744ae40725254f4baa592ed05c912e.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
