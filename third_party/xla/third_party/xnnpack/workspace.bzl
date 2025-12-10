"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "7480edcb300368d5516b583d6312b596cd8c23395c214bb786ec2a1e09eb6b4b",
        strip_prefix = "XNNPACK-dc05a09f076534ce56c6f5b82a0327850c66bf3c",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/dc05a09f076534ce56c6f5b82a0327850c66bf3c.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
