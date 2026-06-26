"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "b1ac2fcb6ed85623430a4ac05ddb08432e3ca87ccf77596ea2b4bc7d5ebad00a",
        strip_prefix = "XNNPACK-23a67314f7afdbb76191589ae090d82bf55afbfa",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/23a67314f7afdbb76191589ae090d82bf55afbfa.zip"),
        patch_file = ["//third_party/xnnpack:cl_clang.patch"],
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
