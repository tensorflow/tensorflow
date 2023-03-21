"""Provides the repo macro to import google benchmark"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports benchmark."""
<<<<<<< HEAD
    BM_COMMIT = "0baacde3618ca617da95375e0af13ce1baadea47"
    BM_SHA256 = "0b921a3bc39e35f4275c8dcc658af2391c150fb966102341287b0401ff2e6f21"
=======
    BM_COMMIT = "f7547e29ccaed7b64ef4f7495ecfff1c9f6f3d03"
    BM_SHA256 = "552ca3d4d1af4beeb1907980f7096315aa24150d6baf5ac1e5ad90f04846c670"
>>>>>>> google_upstream/master
    tf_http_archive(
        name = "com_google_benchmark",
        sha256 = BM_SHA256,
        strip_prefix = "benchmark-{commit}".format(commit = BM_COMMIT),
        build_file = "//third_party/benchmark:benchmark.BUILD",
        urls = tf_mirror_urls("https://github.com/google/benchmark/archive/{commit}.tar.gz".format(commit = BM_COMMIT)),
    )
