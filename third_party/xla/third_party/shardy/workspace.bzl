"""Provides the repository macro to import Shardy."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
<<<<<<< HEAD
    SHARDY_COMMIT = "e99cfa73916a8758e35015d61767ba7e986fc79d"
    SHARDY_SHA256 = "973ba2b5c77337157e37ff331a111873ac6eddf9831555b3c18391e910673a6d"
=======
    SHARDY_COMMIT = "7fed8c5400d6ffd7e46e9ae318b7f55a5f67b91a"
    SHARDY_SHA256 = "eb1ce21c6302403d4d215c6f5be24a8790475c861e7303ead2728ab66edeef6d"
>>>>>>> upstream/master

    tf_http_archive(
        name = "shardy",
        sha256 = SHARDY_SHA256,
        strip_prefix = "shardy-{commit}".format(commit = SHARDY_COMMIT),
        urls = tf_mirror_urls("https://github.com/openxla/shardy/archive/{commit}.zip".format(commit = SHARDY_COMMIT)),
        patch_file = ["//third_party/shardy:temporary.patch"],
    )
