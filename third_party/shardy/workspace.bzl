"""Provides the repository macro to import Shardy."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
<<<<<<< HEAD
    SHARDY_COMMIT = "8cf632f5e74c3777fc47d19525a9e2e1c597fc54"
    SHARDY_SHA256 = "12be33b73a5d2bb45df138fab2400b411e4e7b3ad5a058acb6ef397b2252c8df"
=======
    SHARDY_COMMIT = "71c3f3293063aa4301d72c9b3a1e15bd568a8ffd"
    SHARDY_SHA256 = "d60c69d1f842c4358586a5eed572818b7320485924b0bdd4fcc4ca2ea367b37c"
>>>>>>> upstream/master

    tf_http_archive(
        name = "shardy",
        sha256 = SHARDY_SHA256,
        strip_prefix = "shardy-{commit}".format(commit = SHARDY_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/shardy/archive/{commit}.zip".format(commit = SHARDY_COMMIT)),
        patch_file = ["//third_party/shardy:temporary.patch"],
    )
