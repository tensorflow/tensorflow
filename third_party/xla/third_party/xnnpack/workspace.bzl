"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "332dcb355b65bd290d275a06506b9eda41d0320e84793230681e1cc74c560685",
        strip_prefix = "XNNPACK-43c2b577ce92ddd4cda925c4d060867c12e70de0",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/43c2b577ce92ddd4cda925c4d060867c12e70de0.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
