"""Provides the repository macro to import CUDA Tile IR."""

load("//third_party:quilt_repo.bzl", "quilt_http_archive")
load("//third_party:repo.bzl", "tf_mirror_urls")

def repo():
    """Imports CUDA Tile IR."""
    CUDA_TILE_COMMIT = "7e8e2e68fa219716103824c01f7303367cf7df8d"
    CUDA_TILE_SHA256 = "2398f88dd0dbcda3732934c8527485c3941d3a6453c98b6c249461e6309ea44c"

    quilt_http_archive(
        name = "cuda_tile",
        build_file = "//third_party/cuda_tile:cuda_tile.BUILD",
        sha256 = CUDA_TILE_SHA256,
        strip_prefix = "cuda-tile-{}".format(CUDA_TILE_COMMIT),
        urls = tf_mirror_urls("https://github.com/NVIDIA/cuda-tile/archive/{}.tar.gz".format(CUDA_TILE_COMMIT)),
        series = "//third_party/cuda_tile:patches/series",
    )
