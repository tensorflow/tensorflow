"""loads the hwloc library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "hwloc",
        urls = tf_mirror_urls("https://download.open-mpi.org/release/hwloc/v2.7/hwloc-2.7.1.tar.gz"),
        sha256 = "4cb0a781ed980b03ad8c48beb57407aa67c4b908e45722954b9730379bc7f6d5",
        strip_prefix = "hwloc-2.7.1",
        build_file = "//third_party/hwloc:hwloc.BUILD",
        system_build_file = "//third_party/hwloc:BUILD.system",
    )
