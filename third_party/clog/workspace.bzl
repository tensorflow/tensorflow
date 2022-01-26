"""Loads the clog library, used by cpuinfo and XNNPACK."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # Attention: TensorFlow Lite CMake build uses these variables, update only the hash contents. 
    CLOG_COMMIT = "d5e37adf1406cf899d7d9ec1d317c47506ccb970"
    CLOG_SHA256 = "3f2dc1970f397a0e59db72f9fca6ff144b216895c1d606f6c94a507c1e53a025"

    tf_http_archive(
        name = "clog",
        strip_prefix = "cpuinfo-d5e37adf1406cf899d7d9ec1d317c47506ccb970",
        sha256 = CLOG_SHA256,
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/{commit}.tar.gz".format(commit = CLOG_COMMIT)),
        build_file = "//third_party/clog:clog.BUILD",
    )
