"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "f49b2004951427694aba5878f7f961d4e92a0a16"
    TRITON_SHA256 = "ff1215c70623e2dac9c005c12017fb5d11d125b31c6861b252219108da708b4b"

    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-{commit}".format(commit = TRITON_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/triton/archive/{commit}.tar.gz".format(commit = TRITON_COMMIT)),
        # For temporary changes which haven't landed upstream yet.
        patch_file = ["third_party/tensorflow/third_party/triton/cl518645628.patch"],
    )
