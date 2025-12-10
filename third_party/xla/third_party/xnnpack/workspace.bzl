"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "9ff5f0631970f3393522e2fb0b882c7cabc44c76f957d257b507f47611e2df47",
        strip_prefix = "XNNPACK-085272364b1e8168a82994296994d9b02444e82a",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/085272364b1e8168a82994296994d9b02444e82a.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
