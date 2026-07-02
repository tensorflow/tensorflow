"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "9e5e18eb5b9dc596f04458566b49ec0dabb8e82550d94d7cc15f020cfdc92e88",
        strip_prefix = "XNNPACK-f0d3a4805f3d8e49996194633adb6ef90f3b1d06",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/f0d3a4805f3d8e49996194633adb6ef90f3b1d06.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
