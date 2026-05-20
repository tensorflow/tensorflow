"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "46087079714a233377614d022bb843697380636536caf8f01af213e3dac9df81",
        strip_prefix = "XNNPACK-d0004f80c78fed80c230045ee83ff34dc55be81a",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/d0004f80c78fed80c230045ee83ff34dc55be81a.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
