"""cuDNN frontend is a C++ API for cuDNN."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cudnn_frontend_archive",
        build_file = "//third_party:cudnn_frontend.BUILD",
        patch_file = ["//third_party:cudnn_frontend_header_fix.patch"],
        sha256 = "257b3b7f8a99abc096094abc9e5011659117b647d55293bcd2c5659f9181b99e",
        strip_prefix = "cudnn-frontend-1.13.0",
        urls = tf_mirror_urls("https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v1.13.0.zip"),
    )
