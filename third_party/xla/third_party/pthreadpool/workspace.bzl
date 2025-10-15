"""pthreadpool is a portable and efficient thread pool implementation."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "pthreadpool",
        sha256 = "bb468cf1c14951f48c802de905631993fd9bcf395f997187e1e0165ffb88668d",
        strip_prefix = "pthreadpool-75b23a57b4e5ad206e8f69b698b5f1df04cf3d32",
        urls = tf_mirror_urls("https://github.com/google/pthreadpool/archive/75b23a57b4e5ad206e8f69b698b5f1df04cf3d32.zip"),
    )
