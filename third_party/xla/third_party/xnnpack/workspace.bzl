"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "1f0da5e4915978806769b194dff12ef98e22f3fdd3bff64cc0c95ede9e80cec9",
        strip_prefix = "XNNPACK-0374903756f122a2859d28d0b8f4d1374aa3ff90",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/0374903756f122a2859d28d0b8f4d1374aa3ff90.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
