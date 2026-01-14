"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "961965b04b0cee7c0ece34bb21dbdf69e483772ae7bdb275a08e6d457ed7e38b",
        strip_prefix = "XNNPACK-2c1a512208d0481d6e6bd87c2bd5e23408febc3e",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/2c1a512208d0481d6e6bd87c2bd5e23408febc3e.zip"),
        patch_file = ["//third_party/xnnpack:layering_check_fix.patch"],
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
