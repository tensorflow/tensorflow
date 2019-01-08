"""Loads the kissfft library, used by TF Lite."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "kissfft",
        strip_prefix = "kissfft-cddf3833fdf24fa84b79be37efdcd348cae0e39c",
        sha256 = "7ba83a3da1636350472e501e3e6c3418df72466990530ea273c05fa7e3dd8635",
        urls = [
            "https://mirror.bazel.build/github.com/mborgerding/kissfft/archive/cddf3833fdf24fa84b79be37efdcd348cae0e39c.tar.gz",
            "https://github.com/mborgerding/kissfft/archive/cddf3833fdf24fa84b79be37efdcd348cae0e39c.tar.gz",
        ],
        build_file = "//third_party/kissfft:BUILD.bazel",
    )
