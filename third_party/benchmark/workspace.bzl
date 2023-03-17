"""Provides the repo macro to import google benchmark"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports benchmark."""
    BM_COMMIT = "4b086c26febc39f4636d82a436fd445b9af9501b"
    BM_SHA256 = "51478d437e82a51f36ca9071fa9bac2e10455bb1ffee85ccf367bd607487ef09"
    tf_http_archive(
        name = "com_google_benchmark",
        sha256 = BM_SHA256,
        strip_prefix = "benchmark-{commit}".format(commit = BM_COMMIT),
        urls = tf_mirror_urls("https://github.com/google/benchmark/archive/{commit}.tar.gz".format(commit = BM_COMMIT)),
    )
