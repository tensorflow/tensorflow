"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "894fecbdc472f194f3d69652b4bae3a8636e9ea6cbd16515b16d33d24d9aa480",
        strip_prefix = "XNNPACK-76de13802d1c1b286b21694734d87f1683767b8f",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/76de13802d1c1b286b21694734d87f1683767b8f.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
