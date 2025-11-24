"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "a633a48ba393211771204d25ebc5f35359b71bfbefaa6e955aa92570caede727",
        strip_prefix = "XNNPACK-fa0fd6471a39a5d66a59d4cd8f8cc4a93a4bd470",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/fa0fd6471a39a5d66a59d4cd8f8cc4a93a4bd470.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
