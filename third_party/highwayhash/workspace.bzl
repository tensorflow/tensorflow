"""loads the highwayhash library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive")

def repo():
    tf_http_archive(
        name = "highwayhash",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/highwayhash/archive/fd3d9af80465e4383162e4a7c5e2f406e82dd968.tar.gz",
            "https://github.com/google/highwayhash/archive/fd3d9af80465e4383162e4a7c5e2f406e82dd968.tar.gz",
        ],
        sha256 = "9c3e0e87d581feeb0c18d814d98f170ff23e62967a2bd6855847f0b2fe598a37",
        strip_prefix = "highwayhash-fd3d9af80465e4383162e4a7c5e2f406e82dd968",
        build_file = "//third_party/highwayhash:BUILD.bazel",
    )
