"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/triton:common/series.bzl", "common_patch_list")
load("//third_party/triton:oss_only/series.bzl", "oss_only_patch_list")

def repo():
    """Imports Triton."""

    TRITON_COMMIT = "609ced5e3f04e55234115524eb734822331a37d7"
    TRITON_SHA256 = "979b9f9fd6a1dc6a69de20f60357c9b9dc0cbfba3b1169280c75351b592e8b05"
    tf_http_archive(
        name = "triton",
        sha256 = TRITON_SHA256,
        strip_prefix = "triton-" + TRITON_COMMIT,
        urls = tf_mirror_urls("https://github.com/triton-lang/triton/archive/{}.tar.gz".format(TRITON_COMMIT)),
        patch_file = common_patch_list + oss_only_patch_list,
    )
