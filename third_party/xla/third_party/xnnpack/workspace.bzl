"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "d36a005c707c0cf26696acfb5ef27d55a37551a49ed2eeb5979815a61138f07d",
        strip_prefix = "XNNPACK-ea1906f8df2faf8172da1b341c563bf9115581dd",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/ea1906f8df2faf8172da1b341c563bf9115581dd.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
