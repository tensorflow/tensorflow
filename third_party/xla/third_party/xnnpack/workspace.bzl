"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "6ebde53e2dc0af6d16e3e2f46f6e9428a76f4356c2e177c5f558c0b4c5cf9e83",
        strip_prefix = "XNNPACK-8388bd78690515166d59f1b28e593a455a41d580",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/8388bd78690515166d59f1b28e593a455a41d580.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
