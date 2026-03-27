"""Provides the repository macro to import Shardy."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    SHARDY_COMMIT = "8f0fa5b0f3a1314cbae1fc3eb3381b8e27b2cd36"
    SHARDY_SHA256 = "2153fbde40213f06f141c69bacb4790b8b28dc1b3ec40fa497d84a0f9a6d4eef"

    tf_http_archive(
        name = "shardy",
        sha256 = SHARDY_SHA256,
        strip_prefix = "shardy-{commit}".format(commit = SHARDY_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/shardy/archive/{commit}.zip".format(commit = SHARDY_COMMIT)),
        patch_file = ["//third_party/shardy:temporary.patch"],
    )
