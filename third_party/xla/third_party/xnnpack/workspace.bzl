"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "630521a16c1ba0ecfca4b5a1557bb61fd97dff2b2e5ca284b709c469f68b8248",
        strip_prefix = "XNNPACK-4f8026d0af017bb21bf670bda8c82a284bd42cde",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/4f8026d0af017bb21bf670bda8c82a284bd42cde.zip"),
        patch_file = ["//third_party/xnnpack:layering_check_fix.patch"],
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
