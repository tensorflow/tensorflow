"""Loads the xprof library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo(**kwargs):
    """Loads the xprof library, used by TF."""
    tf_http_archive(
        name = "org_xprof",
        sha256 = "51fc9986e8b4a7d2f6333d914454e333fc955076de06d06826ae6babf02e6310",
        strip_prefix = "xprof-c2e053062acdcdf2119eb1ab89d19b4ac6ebe9fc",
        patch_file = ["//third_party/xprof:xprof.patch"],
        urls = tf_mirror_urls("https://github.com/openxla/xprof/archive/c2e053062acdcdf2119eb1ab89d19b4ac6ebe9fc.zip"),
        **kwargs
    )
    tf_http_archive(
        name = "rules_android",
        sha256 = "fe3d8c4955857b44019d83d05a0b15c2a0330a6a0aab990575bb397e9570ff1b",
        strip_prefix = "rules_android-0.6.0-alpha1",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_android/releases/download/v0.6.0-alpha1/rules_android-v0.6.0-alpha1.tar.gz"),
    )
    tf_http_archive(
        name = "perfetto",
        sha256 = "b25023f3281165a1a7d7cde9f3ed2dfcfce022ffd727e77f6589951e0ba6af9a",
        strip_prefix = "perfetto-53.0",
        urls = tf_mirror_urls("https://github.com/google/perfetto/archive/refs/tags/v53.0.tar.gz"),
    )
    tf_http_archive(
        name = "perfetto_cfg",
        patch_cmds = ['echo \'exports_files(["perfetto_cfg.bzl"])\' > BUILD.bazel'],
        sha256 = "b25023f3281165a1a7d7cde9f3ed2dfcfce022ffd727e77f6589951e0ba6af9a",
        strip_prefix = "perfetto-53.0/bazel/standalone",
        urls = tf_mirror_urls("https://github.com/google/perfetto/archive/refs/tags/v53.0.tar.gz"),
    )
