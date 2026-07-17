"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "0dc79da84559117ed446db6798aea3e166b5da85a841f723787b234ae2d34b0a",
        strip_prefix = "XNNPACK-efa2e754e390ab3024f6aa617385cb28bdfa9969",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/efa2e754e390ab3024f6aa617385cb28bdfa9969.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
