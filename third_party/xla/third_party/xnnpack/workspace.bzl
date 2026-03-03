"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "264af70e73c677ecb6e3ba1e2ce956654388af9660418961049ac1573dc4d0c5",
        strip_prefix = "XNNPACK-26feacc3ebe30d1f3a82e1ff3938ef9859d74640",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/26feacc3ebe30d1f3a82e1ff3938ef9859d74640.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
