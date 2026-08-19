"""Loads libdrm headers for ROCm compatibility."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Import libdrm headers."""

    # libdrm 2.4.120 - a recent stable version
    tf_http_archive(
        name = "libdrm",
        build_file = str(Label("//third_party/libdrm:libdrm.BUILD")),
        sha256 = "3bf55363f76c7250946441ab51d3a6cc0ae518055c0ff017324ab76cdefb327a",
        strip_prefix = "libdrm-2.4.120",
        urls = tf_mirror_urls(
            "https://dri.freedesktop.org/libdrm/libdrm-2.4.120.tar.xz",
        ),
    )
