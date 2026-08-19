"""pthreadpool is a portable and efficient thread pool implementation."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "pthreadpool",
        sha256 = "5ab4e8f63e3dcf62048360c216532bdf62f00dc204883a52d91230402f0feb6a",
        strip_prefix = "pthreadpool-02460584c6092e527c8b89f7df4de143d70e801f",
        urls = tf_mirror_urls("https://github.com/google/pthreadpool/archive/02460584c6092e527c8b89f7df4de143d70e801f.zip"),
    )
