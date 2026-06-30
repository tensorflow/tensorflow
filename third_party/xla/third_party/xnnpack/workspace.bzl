"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "00358672af9e88da8dc1394f75e59291260568cce9568e40e9c51c9483554468",
        strip_prefix = "XNNPACK-c2e81f01b01fca3327d4b3aa070b56085f2603bd",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/c2e81f01b01fca3327d4b3aa070b56085f2603bd.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
