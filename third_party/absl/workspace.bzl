"""Provides the repository macro to import absl."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    """Imports absl."""

    # Attention: tools parse and update these lines.
    ABSL_COMMIT = "f1dad1e9b277066d676034d8f2a982b9e64310de"
    ABSL_SHA256 = "0095ed54f29c629d4a35812597c8e811b73793d852d51cb1b5c00904c3eb3976"

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
