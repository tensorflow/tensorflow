"""Provides the repository macro to import spdlog."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports spdlog."""

    SPDLOG_VERSION = "1.15.2"
    SPDLOG_SHA256 = "7a80896357f3e8e920e85e92633b14ba0f229c506e6f978578bdc35ba09e9a5d"

    tf_http_archive(
        name = "spdlog",
        sha256 = SPDLOG_SHA256,
        strip_prefix = "spdlog-{version}".format(version = SPDLOG_VERSION),
        urls = tf_mirror_urls("https://github.com/gabime/spdlog/archive/refs/tags/v{version}.tar.gz".format(version = SPDLOG_VERSION)),
        build_file = "//third_party/spdlog:spdlog.BUILD",
        patch_file = ["//third_party/spdlog:smoke_test.cc.patch"],
    )
