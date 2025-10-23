"""pthreadpool is a portable and efficient thread pool implementation."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "pthreadpool",
        sha256 = "f602ab141bdc5d5872a79d6551e9063b5bfa7ad6ad60cceaa641de5c45c86d70",
        strip_prefix = "pthreadpool-0e6ca13779b57d397a5ba6bfdcaa8a275bc8ea2e",
        urls = tf_mirror_urls("https://github.com/google/pthreadpool/archive/0e6ca13779b57d397a5ba6bfdcaa8a275bc8ea2e.zip"),
    )
