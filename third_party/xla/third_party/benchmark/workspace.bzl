"""Provides the repo macro to import google benchmark"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports benchmark."""

    # v1.9.5
    BM_COMMIT = "192ef10025eb2c4cdd392bc502f0c852196baa48"
    BM_SHA256 = "f82705a2726d8f6cdcda274b841f6314dbfc6f731cdda06c946f310ec1cc3ad9"
    tf_http_archive(
        name = "com_google_benchmark",
        sha256 = BM_SHA256,
        strip_prefix = "benchmark-{commit}".format(commit = BM_COMMIT),
        urls = tf_mirror_urls("https://github.com/google/benchmark/archive/{commit}.tar.gz".format(commit = BM_COMMIT)),
    )
