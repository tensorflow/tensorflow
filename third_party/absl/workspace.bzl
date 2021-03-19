"""Provides the repository macro to import absl."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports absl."""

    # Attention: tools parse and update these lines.
    ABSL_COMMIT = "b56cbdd23834a65682c0b46f367f8679e83bc894"
    ABSL_SHA256 = "6eb8a1c4e29a1e1fa9f13eacb493fbc5c40d044658fe99eb46afb6297c5760f2"

    tf_http_archive(
        name = "com_google_absl",
        sha256 = ABSL_SHA256,
        build_file = "//third_party/absl:com_google_absl.BUILD",
        # TODO(mihaimaruseac): Remove the patch when https://github.com/abseil/abseil-cpp/issues/326 is resolved
        patch_file = "//third_party/absl:com_google_absl_fix_mac_and_nvcc_build.patch",
        strip_prefix = "abseil-cpp-{commit}".format(commit = ABSL_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/{commit}.tar.gz".format(commit = ABSL_COMMIT),
            "https://github.com/abseil/abseil-cpp/archive/{commit}.tar.gz".format(commit = ABSL_COMMIT),
        ],
    )
