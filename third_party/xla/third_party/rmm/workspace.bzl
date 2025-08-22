"""Provides the repository macro to import rmm."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports rmm."""

    RMM_VERSION = "25.08.00"
    RMM_SHA256 = "6931f4de923b617af8c3b97505d79fd3b7b6b5492c1b5a8cd8bcfdc147cdf458"

    tf_http_archive(
        name = "rmm",
        sha256 = RMM_SHA256,
        strip_prefix = "rmm-{version}".format(version = RMM_VERSION),
        urls = tf_mirror_urls("https://github.com/rapidsai/rmm/archive/refs/tags/v{version}.tar.gz".format(version = RMM_VERSION)),
        build_file = "//third_party/rmm:rmm.BUILD",
        patch_file = ["//third_party/rmm:logger_macros.hpp.patch"],
    )
