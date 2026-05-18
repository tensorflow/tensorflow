"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "93a18bafec071b098a15505324999e49a7c72edb6cd129ad43a295477b6f5acc",
        strip_prefix = "XNNPACK-98c8ded4369968fab823ebbef877bfcdd87beb58",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/98c8ded4369968fab823ebbef877bfcdd87beb58.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
