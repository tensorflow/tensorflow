"""Provides the repository macro to import gutil."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports gutil."""

    # Attention: tools parse and update these lines.
    GUTIL_COMMIT = "b498c8d364ac96c32194f71f8f719707a398e82b"  # LTS 20250502.0
    GUTIL_SHA256 = "aeca39e4a50f9607437731aba79189a64ff51b742c00f8b80049686e7600e09f"

    tf_http_archive(
        name = "com_google_gutil",
        sha256 = GUTIL_SHA256,
        strip_prefix = "gutil-{commit}".format(commit = GUTIL_COMMIT),
        urls = tf_mirror_urls("https://github.com/google/gutil/archive/{commit}.tar.gz".format(commit = GUTIL_COMMIT)),
    )
