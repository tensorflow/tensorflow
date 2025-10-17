"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "6c9697b8def543d5dd678e04c03ebdc73a29065dd75ec7df1eec28b3992df507",
        strip_prefix = "XNNPACK-ec363e32757a9e0af0c6be5fceeda31c4fd00451",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/ec363e32757a9e0af0c6be5fceeda31c4fd00451.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
