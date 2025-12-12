"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "2d5e0b17d2c25c7100f66e58e7d76b9c4b8a65b1d86c33c9214dc05fce00ee69",
        strip_prefix = "XNNPACK-6400256d3a687d52ae268a553d7208534f39800a",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/6400256d3a687d52ae268a553d7208534f39800a.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
