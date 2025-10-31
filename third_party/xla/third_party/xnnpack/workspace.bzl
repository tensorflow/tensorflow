"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "418fd65e877737a7cb56127ec57f4e741671b5d11a1fa0add41c2e15934c1a90",
        strip_prefix = "XNNPACK-ac8a153cdf64f9c70246fd3550cb7338e789c69e",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/ac8a153cdf64f9c70246fd3550cb7338e789c69e.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
