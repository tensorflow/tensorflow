"""Provides the repository macro to import gemmlowp."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports gemmlowp."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    GEMMLOWP_COMMIT = "16e8662c34917be0065110bfcd9cc27d30f52fdf"
    GEMMLOWP_SHA256 = "c3feb896a1b42595cf9a508ed64ed0dc3cd84fffdb8eed790d02d0534ab322ce"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/gemmlowp.cmake)

    tf_http_archive(
        name = "gemmlowp",
        sha256 = GEMMLOWP_SHA256,
        strip_prefix = "gemmlowp-{commit}".format(commit = GEMMLOWP_COMMIT),
        urls = tf_mirror_urls("https://github.com/google/gemmlowp/archive/{commit}.zip".format(commit = GEMMLOWP_COMMIT)),
    )
