"""Provides the repository macro to import absl."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports absl."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    ABSL_COMMIT = "d38452e1ee03523a208362186fd42248ff2609f6"  # LTS 20250814.1
    ABSL_SHA256 = "d1abe9da2003e6cbbd7619b0ced3e52047422f4f4ac6c66a9bef5d2e99fea837"
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
            "//third_party/absl:rules_cc.patch",
        ],
        repo_mapping = {
            "@google_benchmark": "@com_google_benchmark",
            "@googletest": "@com_google_googletest",
        },
    )
