"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "366c50006fcc3616ecacb14c20e7c7ac111f3086b86864d3378eceb07610e1a7",
        strip_prefix = "XNNPACK-bf362487d3c00ed7040a0c9c4b885f2ba20a6c45",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/bf362487d3c00ed7040a0c9c4b885f2ba20a6c45.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
