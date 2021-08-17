"""Provides the repo macro to import google benchmark"""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports benchmark."""
    BM_COMMIT = "64cb55e91067860548cb95e012a38f2e5b71e026"
    BM_SHA256 = "480bb4f1ffa402e5782a20dc8986f5c86b87c497195dc53c9067e502ff45ef57"
    tf_http_archive(
        name = "com_google_benchmark",
        sha256 = BM_SHA256,
        strip_prefix = "benchmark-{commit}".format(commit = BM_COMMIT),
        build_file = "//third_party/benchmark:BUILD.bazel",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/benchmark/archive/{commit}.tar.gz".format(commit = BM_COMMIT),
            "https://github.com/google/benchmark/archive/{commit}.tar.gz".format(commit = BM_COMMIT),
        ],
    )
