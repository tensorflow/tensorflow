"""Loads the xprof library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo(**kwargs):
    """Loads the xprof library, used by TF."""
    tf_http_archive(
        name = "org_xprof",
        sha256 = "ce56344e21802f90b90ec74f3b800859110299f58ca2a25e5fe739e3257229e4",
        strip_prefix = "xprof-17c8c27e0af348674e60e22c327c036c78393ed4",
        urls = tf_mirror_urls("https://github.com/openxla/xprof/archive/17c8c27e0af348674e60e22c327c036c78393ed4.zip"),
        **kwargs
    )
