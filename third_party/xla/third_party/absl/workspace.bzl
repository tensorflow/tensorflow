"""Provides the repository macro to import absl."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports absl."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    ABSL_COMMIT = "54fac219c4ef0bc379dfffb0b8098725d77ac81b"  # LTS 20240116.3
    ABSL_SHA256 = "a862ce94f77979ce36d2ca21ad3ca36b60838083392247b301b085a06d9f2b1a"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/abseil-cpp.cmake)

    tf_http_archive(
        name = "com_google_absl",
        sha256 = ABSL_SHA256,
        strip_prefix = "abseil-cpp-{commit}".format(commit = ABSL_COMMIT),
        urls = tf_mirror_urls("https://github.com/abseil/abseil-cpp/archive/{commit}.tar.gz".format(commit = ABSL_COMMIT)),
        patch_file = [
            "//third_party/absl:nvidia_jetson.patch",
            "//third_party/absl:build_dll.patch",
            "//third_party/absl:nullability_macros.patch",
        ],
    )
