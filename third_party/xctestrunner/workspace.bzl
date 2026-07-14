"""Loads the xctestrunner library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "xctestrunner",
        strip_prefix = "xctestrunner-4c5709da9444eae6bba2425734b8654635bed0a6",
        sha256 = "e5d4c53c3965ae943fb08ccd7df0efd75590213fce5052388f23fad81a649f5a",
        urls = tf_mirror_urls("https://github.com/google/xctestrunner/archive/4c5709da9444eae6bba2425734b8654635bed0a6.tar.gz"),
    )
