"""Provides the repo macro to import google benchmark"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports benchmark."""
    BM_COMMIT = "0baacde3618ca617da95375e0af13ce1baadea47"
    BM_SHA256 = "0b921a3bc39e35f4275c8dcc658af2391c150fb966102341287b0401ff2e6f21"
    tf_http_archive(
        name = "com_google_benchmark",
        sha256 = BM_SHA256,
        strip_prefix = "benchmark-{commit}".format(commit = BM_COMMIT),
        build_file = "//third_party/benchmark:benchmark.BUILD",
        urls = tf_mirror_urls("https://github.com/google/benchmark/archive/{commit}.tar.gz".format(commit = BM_COMMIT)),
    )
