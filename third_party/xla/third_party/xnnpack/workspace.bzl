"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "b75c85b77e2d20b710763978c00385b27869f28a5f0a4967050c6d06767043ce",
        strip_prefix = "XNNPACK-2dbaa1cd9faac161a59f4e1f3d0835991e2370d9",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/2dbaa1cd9faac161a59f4e1f3d0835991e2370d9.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
