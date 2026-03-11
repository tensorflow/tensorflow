"""Provides the repository macro to import xprof."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo(repo_mapping = None):
    """Imports xprof."""

    # v2.21.4
    tf_http_archive(
        name = "org_xprof",
        sha256 = "1e70d4e3794888073647f8c5461cb84a3fd28578accfdb1704e7a0dd56b0ccbb",
        strip_prefix = "xprof-bb565c5b9d990912771192b9004331a57c4e4174",
        urls = tf_mirror_urls("https://github.com/openxla/xprof/archive/bb565c5b9d990912771192b9004331a57c4e4174.zip"),
        repo_mapping = repo_mapping,
    )
