"""Provides the repository macro to import absl."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports absl."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    ABSL_COMMIT = "d407ef122a08203648451e0fec77b3f868b71112"  # LTS 20260107.0
    ABSL_SHA256 = "1d294b5e3cd8b19e63eb047d635953a33e553c47c647524e9e78428692f7d53c"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/abseil-cpp.cmake)

    tf_http_archive(
        name = "com_google_absl",
        sha256 = ABSL_SHA256,
        strip_prefix = "abseil-cpp-{commit}".format(commit = ABSL_COMMIT),
        urls = tf_mirror_urls("https://github.com/abseil/abseil-cpp/archive/{commit}.tar.gz".format(commit = ABSL_COMMIT)),
        patch_file = [
            "//third_party/absl:btree.patch",
            "//third_party/absl:build_dll.patch",
            "//third_party/absl:endian.patch",
        ],
        repo_mapping = {
            "@google_benchmark": "@com_google_benchmark",
            "@googletest": "@com_google_googletest",
        },
    )
