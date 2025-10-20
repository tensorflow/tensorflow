"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "a4f998ed5557e85fb459da22d753d1b9b9ba7aa7cc844c00b0d159a390185a44",
        strip_prefix = "XNNPACK-d496777428ca92c3080d58e1a0f16e37e3cf9752",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/d496777428ca92c3080d58e1a0f16e37e3cf9752.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
