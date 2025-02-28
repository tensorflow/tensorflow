"""Provides the repository macro to import Shardy."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    SHARDY_COMMIT = "f7c7663730da5d3d992123d95767ef25a49472a7"
    SHARDY_SHA256 = "bdd18afde09a57bebd70de2403c8b388da4acc7bec17e7c63e65b8d48b8cdfa4"

    tf_http_archive(
        name = "shardy",
        sha256 = SHARDY_SHA256,
        strip_prefix = "shardy-{commit}".format(commit = SHARDY_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/shardy/archive/{commit}.zip".format(commit = SHARDY_COMMIT)),
        patch_file = ["//third_party/shardy:temporary.patch"],
    )
