"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "a23040b6307b67da2319de292a0bb4b39d0e5913fae50a90f955eafa1acb81c7",
        strip_prefix = "XNNPACK-da9a34d9bb68f339c35d2da480ab0734b0a26429",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/da9a34d9bb68f339c35d2da480ab0734b0a26429.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
