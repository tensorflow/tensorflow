"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/triton:common/series.bzl", "common_patch_list")
load("//third_party/triton:oss_only/series.bzl", "oss_only_patch_list")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "c60fd50377344ebce1f3102cf14211189b69184f"
    TRITON_SHA256 = "1b2ff85f16bbd0db2c65344a338b70b1648d6ba2826ef0f863408bc1805b2e67"
    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-" + TRITON_COMMIT,
        urls = tf_mirror_urls("https://github.com/triton-lang/triton/archive/{}.tar.gz".format(TRITON_COMMIT)),
        patch_file = common_patch_list + oss_only_patch_list,
    )
