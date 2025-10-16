"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "3bdfafcd684e765b9dd2e2cc08a68afdade086cf966e6319c1bde779838feae8",
        strip_prefix = "XNNPACK-f9b86a81a8f75a2b938af243d6edbd2feacc391b",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/f9b86a81a8f75a2b938af243d6edbd2feacc391b.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
