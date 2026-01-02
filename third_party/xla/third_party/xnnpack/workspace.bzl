"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "44bf8a258cfd0d7b500b6058a2bb5c7387c8cebba295cfca985a68d16513f7c8",
        strip_prefix = "XNNPACK-25b42dfddb0ee22170d73ff0d4b333ea1e6edfeb",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/25b42dfddb0ee22170d73ff0d4b333ea1e6edfeb.zip"),
        patch_file = ["//third_party/xnnpack:layering_check_fix.patch"],
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
