"""Provides the repository macro to import gemmlowp."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports gemmlowp."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    GEMMLOWP_COMMIT = "e844ffd17118c1e17d94e1ba4354c075a4577b88"
    GEMMLOWP_SHA256 = "522b7a82d920ebd0c4408a5365866a40b81d1c0d60b2369011d315cca03c6476"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/gemmlowp.cmake)

    tf_http_archive(
        name = "gemmlowp",
        sha256 = GEMMLOWP_SHA256,
        strip_prefix = "gemmlowp-{commit}".format(commit = GEMMLOWP_COMMIT),
        urls = tf_mirror_urls("https://github.com/google/gemmlowp/archive/{commit}.zip".format(commit = GEMMLOWP_COMMIT)),
    )
