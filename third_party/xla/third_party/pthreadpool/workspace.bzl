"""pthreadpool is a portable and efficient thread pool implementation."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "pthreadpool",
        sha256 = "8b1d13195842c9b7e8ef5aa7d9b44ca4168a41b8ae97b4e50db4fcc562211f5b",
        strip_prefix = "pthreadpool-d561aae9dfeab38ff595a0ae3e6bbd90b862c5f8",
        urls = tf_mirror_urls("https://github.com/google/pthreadpool/archive/d561aae9dfeab38ff595a0ae3e6bbd90b862c5f8.zip"),
    )
