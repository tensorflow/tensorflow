"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "97a53ebb239e5835cc7ba24f1aa032283f0c6fe4f963977daad29a83ad75d9a0",
        strip_prefix = "XNNPACK-262e13f37afdef03ed1b8ad7d99055b42731307a",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/262e13f37afdef03ed1b8ad7d99055b42731307a.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
