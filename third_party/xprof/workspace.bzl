"""Loads the xprof library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo(**kwargs):
    """Loads the xprof library, used by TF."""
    tf_http_archive(
        name = "org_xprof",
        sha256 = "4a9c4401c106f3a5dfb5eb481dadf614f567a6e7927e138f2cbe4afaaeed3fd8",
        strip_prefix = "xprof-01b4072213efa05e26b7e3e18f10f5a5a7a13975",
        urls = tf_mirror_urls("https://github.com/openxla/xprof/archive/01b4072213efa05e26b7e3e18f10f5a5a7a13975.zip"),
        **kwargs
    )
