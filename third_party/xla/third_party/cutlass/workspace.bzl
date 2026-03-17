load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "cutlass_archive",
        build_file = "//third_party:cutlass.BUILD",
        sha256 = "a7739ca3dc74e3a5cb57f93fc95224c5e2a3c2dff2c16bb09a5e459463604c08",
        strip_prefix = "cutlass-3.8.0",
        urls = tf_mirror_urls("https://github.com/NVIDIA/cutlass/archive/refs/tags/v3.8.0.zip"),
    )
