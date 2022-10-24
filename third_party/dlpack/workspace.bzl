"""DLPack is a protocol for sharing arrays between deep learning frameworks."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "dlpack",
        strip_prefix = "dlpack-9351cf542ab478499294864ff3acfdab5c8c5f3d",
        sha256 = "7aca112f2809b7e9523e9b47b04a393affeca38247861951f07c42dee10180e2",
        urls = tf_mirror_urls("https://github.com/dmlc/dlpack/archive/9351cf542ab478499294864ff3acfdab5c8c5f3d.tar.gz"),
        build_file = "//third_party/dlpack:dlpack.BUILD",
    )
