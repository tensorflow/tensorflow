"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "4fc18b4cbd2c77736ab24f7eaa6a781f56fc53bdb28b9d2b843b9dfe1d7293ff",
        strip_prefix = "XNNPACK-0360bf046f6f33e05da7d58c8e0cb7ac3974457a",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/0360bf046f6f33e05da7d58c8e0cb7ac3974457a.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
