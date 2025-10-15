"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "a0632be61cfbe44b8dbba4502d489e4721b25c66be04a1a75f1912d1e4836020",
        strip_prefix = "XNNPACK-5adfd1d0ea071a7e1ca33d15ffdbbe183005b3f0",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/5adfd1d0ea071a7e1ca33d15ffdbbe183005b3f0.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
