"""Provides the repository macro to import absl."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports absl."""

    # Attention: tools parse and update these lines.
    ABSL_COMMIT = "df3ea785d8c30a9503321a3d35ee7d35808f190d"
    ABSL_SHA256 = "f368a8476f4e2e0eccf8a7318b98dafbe30b2600f4e3cf52636e5eb145aba06a"

    tf_http_archive(
        name = "com_google_absl",
        sha256 = ABSL_SHA256,
        build_file = "//third_party/absl:com_google_absl.BUILD",
        # TODO: Remove the patch when https://github.com/abseil/abseil-cpp/issues/326 is resolved
        # and when TensorFlow is build against CUDA 10.2
        patch_file = "//third_party/absl:com_google_absl_fix_mac_and_nvcc_build.patch",
        strip_prefix = "abseil-cpp-{commit}".format(commit = ABSL_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/{commit}.tar.gz".format(commit = ABSL_COMMIT),
            "https://github.com/abseil/abseil-cpp/archive/{commit}.tar.gz".format(commit = ABSL_COMMIT),
        ],
    )
