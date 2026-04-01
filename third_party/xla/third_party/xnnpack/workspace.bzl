"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "ec57e0dbeed53e9c0cb2334346688ebd522b36ad326f4a3f2f86a701179fbdb7",
        strip_prefix = "XNNPACK-132a7b041d8f8ef74172d3a8cd509d6751fdf33d",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/132a7b041d8f8ef74172d3a8cd509d6751fdf33d.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
