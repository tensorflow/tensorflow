"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "13ae01126b6d4a8b6769433c2a942d6204a3f97157d9c83d79cbfeec1041398c",
        strip_prefix = "XNNPACK-53a1797ba4360cbde068f2a984652be0f0b7b6fe",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/53a1797ba4360cbde068f2a984652be0f0b7b6fe.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
