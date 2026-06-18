"""Provides the repository macro to import absl."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports absl."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    ABSL_COMMIT = "255c84dadd029fd8ad25c5efb5933e47beaa00c7"  # LTS 20260107.1
    ABSL_SHA256 = "87e91fb785a2d0233f4599317afd576b7736e6732d557bdcdfdc11990bd333ef"
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
            "//third_party/absl:append_and_overwrite.patch",
        ],
        repo_mapping = {
            "@google_benchmark": "@com_google_benchmark",
            "@googletest": "@com_google_googletest",
        },
    )
