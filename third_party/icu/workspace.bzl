"""Loads a lightweight subset of the ICU library for Unicode processing."""

load("//third_party:repo.bzl", "third_party_http_archive")

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def repo():
    third_party_http_archive(
        name = "icu",
        strip_prefix = "icu-release-62-1",
        sha256 = "e15ffd84606323cbad5515bf9ecdf8061cc3bf80fb883b9e6aa162e485aa9761",
        urls = [
            "https://mirror.bazel.build/github.com/unicode-org/icu/archive/release-62-1.tar.gz",
            "https://github.com/unicode-org/icu/archive/release-62-1.tar.gz",
        ],
        build_file = "//third_party/icu:BUILD.bazel",
        system_build_file = "//third_party/icu:BUILD.system",
        patch_file = clean_dep("//third_party/icu:udata.patch"),
    )
