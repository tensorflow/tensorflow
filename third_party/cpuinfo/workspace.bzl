"""Loads the cpuinfo library, used by XNNPACK."""

load("//third_party:repo.bzl", "third_party_http_archive")

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def repo():
    third_party_http_archive(
        name = "cpuinfo",
        strip_prefix = "cpuinfo-6cecd15784fcb6c5c0aa7311c6248879ce2cb8b2",
        sha256 = "b1f2ee97e46d8917a66bcb47452fc510d511829556c93b83e06841b9b35261a5",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pytorch/cpuinfo/archive/6cecd15784fcb6c5c0aa7311c6248879ce2cb8b2.zip",
            "https://github.com/pytorch/cpuinfo/archive/6cecd15784fcb6c5c0aa7311c6248879ce2cb8b2.zip",
        ],
        build_file = "//third_party/cpuinfo:BUILD.bazel",
        patch_file = clean_dep("//third_party/cpuinfo:cpuinfo.patch"),
    )
