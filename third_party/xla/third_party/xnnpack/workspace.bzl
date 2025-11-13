"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "027376a71384311a0ddca0fc986dea621bac0f8b30c96365bf4d2937b627226f",
        strip_prefix = "XNNPACK-decc685b0ecfd00da5a2168eb03b0c795678f084",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/decc685b0ecfd00da5a2168eb03b0c795678f084.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
