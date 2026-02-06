"""loads the re2 library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "com_googlesource_code_re2",
        sha256 = "8635bc46ac8d73974b4198229805287c8d620245f2081af155d7d96d4988a3a5",
        strip_prefix = "re2-927f5d53caf8111721e734cf24724686bb745f55",
        urls = tf_mirror_urls("https://github.com/google/re2/archive/927f5d53caf8111721e734cf24724686bb745f55.tar.gz"),
    )
