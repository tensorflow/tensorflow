"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "447380ee8ed6e10a475f4754321f4ca83c86d6a42eb1d7172f5aafe10a907034",
        strip_prefix = "XNNPACK-85ba6247ad2168015ab083eaca1d72279b6f8c39",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/85ba6247ad2168015ab083eaca1d72279b6f8c39.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
