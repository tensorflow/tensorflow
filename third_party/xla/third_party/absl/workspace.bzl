"""Provides the repository macro to import absl."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports absl."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    ABSL_COMMIT = "5650e9cf76d3be4318d5fa3af38ee483ddfd5e4a"  # LTS 20260526.0
    ABSL_SHA256 = "bac569e31429f85d7bdd9f90e4767442faf9c8fa1def08a66a7d5d78e0fba204"
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
