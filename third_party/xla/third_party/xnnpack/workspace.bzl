"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "fb11a1c58fcfd1512dbf0cdc1d0ee855f443b9f9de91fb5033fd5c219f92f11a",
        strip_prefix = "XNNPACK-ae746db8255aa93704012a98b4b030eefd17357d",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/ae746db8255aa93704012a98b4b030eefd17357d.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
