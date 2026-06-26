"""Loads the xprof library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo(**kwargs):
    """Loads the xprof library, used by TF."""
    tf_http_archive(
        name = "org_xprof",
        sha256 = "7aeec1b4c1d10e1587a932cc6bdc6936956b36efc89f5c8c1e4e2cb3d3b50fde",
        strip_prefix = "xprof-xprof-v2.22.3",
        patch_file = ["//third_party/xprof:xprof.patch"],
        urls = tf_mirror_urls("https://github.com/openxla/xprof/archive/refs/tags/xprof-v2.22.3.zip"),
        **kwargs
    )
