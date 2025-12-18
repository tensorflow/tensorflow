"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "08976c0ba6495775f78d738adbcc60a567b5826774f23d3c403486c70ff79772",
        strip_prefix = "XNNPACK-183297df5c945236cbc4bb1f625f9f2008bfc564",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/183297df5c945236cbc4bb1f625f9f2008bfc564.zip"),
        patch_file = ["//third_party/xnnpack:layering_check_fix.patch"],
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
