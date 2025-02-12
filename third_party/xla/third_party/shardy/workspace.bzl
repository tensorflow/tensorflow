"""Provides the repository macro to import Shardy."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    SHARDY_COMMIT = "6734e22c31ad8e59d18a647e86e6e43c40c2356a"
    SHARDY_SHA256 = "db169e1ef23e4cd8b6b7c2773f28b13430556e48a15348e177832cca8f59d784"

    tf_http_archive(
        name = "shardy",
        sha256 = SHARDY_SHA256,
        strip_prefix = "shardy-{commit}".format(commit = SHARDY_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/shardy/archive/{commit}.zip".format(commit = SHARDY_COMMIT)),
        patch_file = ["//third_party/shardy:temporary.patch"],
    )
