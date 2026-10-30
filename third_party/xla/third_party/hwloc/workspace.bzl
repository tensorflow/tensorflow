"""loads the hwloc library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "hwloc",
        urls = tf_mirror_urls("https://download.open-mpi.org/release/hwloc/v2.13/hwloc-2.13.0.tar.gz"),
        sha256 = "1514a5253f0a5c23bc006d3bdd30a6f6125c9a8dc9b5fa4984913d1fff45315d",
        strip_prefix = "hwloc-2.13.0",
        build_file = "//third_party/hwloc:hwloc.BUILD",
        system_build_file = "//third_party/hwloc:BUILD.system",
    )
