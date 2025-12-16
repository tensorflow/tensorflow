"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "a89879422c6da8240cffb8ff67f5cd11f0362cb2a174ee9cd96b450e53902ca3",
        strip_prefix = "XNNPACK-77468446ebfd9baab7fc4349c32608c9675cf6d9",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/77468446ebfd9baab7fc4349c32608c9675cf6d9.zip"),
        patch_file = ["//third_party/xnnpack:layering_check_fix.patch"],
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
