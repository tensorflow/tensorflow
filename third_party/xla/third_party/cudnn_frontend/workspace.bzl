"""cuDNN frontend is a C++ API for cuDNN."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cudnn_frontend_archive",
        build_file = "//third_party:cudnn_frontend.BUILD",
        patch_file = ["//third_party:cudnn_frontend_header_fix.patch"],
        sha256 = "453d4650e6a25ede58fbbd7077c64ebe92734218d474ec7371bb13fa6d2181fa",
        strip_prefix = "cudnn-frontend-1.16.1",
        urls = tf_mirror_urls("https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v1.16.1.zip"),
    )
