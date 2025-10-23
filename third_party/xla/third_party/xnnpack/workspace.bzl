"""XNNPACK is a highly optimized library of floating-point neural network inference operators for ARM, WebAssembly, and x86 platforms."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "e03c64b59e633944026356524f3862bb7a3809200a76cdd2bfb98758bd0378ee",
        strip_prefix = "XNNPACK-2f108e3efda443ac9f233671cfacf2b3183b7e94",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/2f108e3efda443ac9f233671cfacf2b3183b7e94.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)
