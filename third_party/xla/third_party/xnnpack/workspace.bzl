"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "a42b56ddad892b70188ce393669217808b57c4d20dc9e685e26d3bcda408b731",
        strip_prefix = "XNNPACK-bccfe73347861f07a0257c0c546c0babcf3257b8",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/bccfe73347861f07a0257c0c546c0babcf3257b8.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
