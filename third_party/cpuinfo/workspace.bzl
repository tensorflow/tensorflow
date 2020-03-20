"""Loads the cpuinfo library, used by XNNPACK."""

load("//third_party:repo.bzl", "third_party_http_archive")

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def repo():
    third_party_http_archive(
        name = "cpuinfo",
        strip_prefix = "cpuinfo-d6c0f915ee737f961915c9d17f1679b6777af207",
        sha256 = "146fc61c3cf63d7d88db963876929a4d373f621fb65568b895efa0857f467770",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pytorch/cpuinfo/archive/d6c0f915ee737f961915c9d17f1679b6777af207.tar.gz",
            "https://github.com/pytorch/cpuinfo/archive/d6c0f915ee737f961915c9d17f1679b6777af207.tar.gz",
        ],
        build_file = "//third_party/cpuinfo:BUILD.bazel",
        patch_file = clean_dep("//third_party/cpuinfo:cpuinfo.patch"),
    )
