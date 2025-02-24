"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "ruy",
        # LINT.IfChange
        sha256 = "70a3ebd5e353293a207c9d54ce9ae004a3b8b2e266fe42ca4fbc0e971e08c375",
        strip_prefix = "ruy-690c14c441387a4ea6e07a9ed89657cec8200b92",
        urls = tf_mirror_urls("https://github.com/google/ruy/archive/690c14c441387a4ea6e07a9ed89657cec8200b92.zip"),
        # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/ruy.cmake)
        build_file = "//third_party/ruy:BUILD",
    )
