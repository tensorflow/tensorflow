"""cpuinfo is a library to detect essential CPU features."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cpuinfo",
        sha256 = "c0254ce97f7abc778dd2df0aaca1e0506dba1cd514fdb9fe88c07849393f8ef4",
        strip_prefix = "cpuinfo-8a9210069b5a37dd89ed118a783945502a30a4ae",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/8a9210069b5a37dd89ed118a783945502a30a4ae.zip"),
    )
