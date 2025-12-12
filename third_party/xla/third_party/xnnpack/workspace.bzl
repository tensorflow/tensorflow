"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "f855387f6c4e7db5facdcd83fc41bc94b1888239b396e055ba48dc6da9d89446",
        strip_prefix = "XNNPACK-e436865104ef12ff872db68ec94ce1c5332a6ecb",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/e436865104ef12ff872db68ec94ce1c5332a6ecb.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
