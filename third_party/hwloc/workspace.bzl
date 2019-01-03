"""loads the hwloc library, used by TF."""

load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    third_party_http_archive(
        name = "hwloc",
        urls = [
            "http://mirror.bazel.build/download.open-mpi.org/release/hwloc/v2.0/hwloc-2.0.3.tar.gz",
            "https://download.open-mpi.org/release/hwloc/v2.0/hwloc-2.0.3.tar.gz",
        ],
        sha256 = "64def246aaa5b3a6e411ce10932a22e2146c3031b735c8f94739534f06ad071c",
        strip_prefix = "hwloc-2.0.3",
        build_file = "//third_party/hwloc:BUILD.bazel",
    )
