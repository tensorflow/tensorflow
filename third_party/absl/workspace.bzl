"""Provides the repository macro to import absl."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports absl."""

    # Attention: tools parse and update these lines.
    ABSL_COMMIT = "6f9d96a1f41439ac172ee2ef7ccd8edf0e5d068c"
    ABSL_SHA256 = "62c27e7a633e965a2f40ff16b487c3b778eae440bab64cad83b34ef1cbe3aa93"

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
