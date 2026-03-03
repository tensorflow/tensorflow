"""Loads the xprof library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo(**kwargs):
    """Loads the xprof library, used by TF."""
    tf_http_archive(
        name = "org_xprof",
        sha256 = "d964d6a8101236f85f7e4a15a7e835ef674c61cc647ca9de3e7a5a1a119bbe56",
        strip_prefix = "xprof-29b42d9060811ea1cf5464aa4368853db8329737",
        urls = tf_mirror_urls("https://github.com/openxla/xprof/archive/29b42d9060811ea1cf5464aa4368853db8329737.zip"),
        **kwargs
    )
