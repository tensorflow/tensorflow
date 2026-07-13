"""pthreadpool is a portable and efficient thread pool implementation."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "pthreadpool",
        sha256 = "00a9a1c633f62290a22ea1db42c4401dffe9f05645fb66d6609ae46a05333a2a",
        strip_prefix = "pthreadpool-9003ee6c137cea3b94161bd5c614fb43be523ee1",
        urls = tf_mirror_urls("https://github.com/google/pthreadpool/archive/9003ee6c137cea3b94161bd5c614fb43be523ee1.zip"),
    )
