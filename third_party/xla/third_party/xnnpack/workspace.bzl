"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "6c25b52b7cac58f5507bc3ac023f582ab0a3d8c96dde3bab90a4a2b727218bc2",
        strip_prefix = "XNNPACK-9d47c4a7fd08370a1d4eb191b6f01244b1240907",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/9d47c4a7fd08370a1d4eb191b6f01244b1240907.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
