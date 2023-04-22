"""Provides the repository macro to import gemmlowp."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports gemmlowp."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    GEMMLOWP_COMMIT = "fda83bdc38b118cc6b56753bd540caa49e570745"
    GEMMLOWP_SHA256 = "43146e6f56cb5218a8caaab6b5d1601a083f1f31c06ff474a4378a7d35be9cfb"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/gemmlowp.cmake)

    tf_http_archive(
        name = "gemmlowp",
        sha256 = GEMMLOWP_SHA256,
        strip_prefix = "gemmlowp-{commit}".format(commit = GEMMLOWP_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/gemmlowp/archive/{commit}.zip".format(commit = GEMMLOWP_COMMIT),
            "https://github.com/google/gemmlowp/archive/{commit}.zip".format(commit = GEMMLOWP_COMMIT),
        ],
    )
