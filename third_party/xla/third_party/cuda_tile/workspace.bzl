"""Provides the repository macro to import CUDA Tile IR."""

load("//third_party:quilt_repo.bzl", "quilt_http_archive")
load("//third_party:repo.bzl", "tf_mirror_urls")

def repo():
    """Imports CUDA Tile IR."""
    CUDA_TILE_COMMIT = "25732e3308abd43c0aa9007a7b3b012598bbcf15"
    CUDA_TILE_SHA256 = "aec27b90ca5b83a654175c9a95470e3bcacd512fea271892dbe76e920d8f98e4"

    quilt_http_archive(
        name = "cuda_tile",
        build_file = "//third_party/cuda_tile:cuda_tile.BUILD",
        sha256 = CUDA_TILE_SHA256,
        strip_prefix = "cuda-tile-{}".format(CUDA_TILE_COMMIT),
        urls = tf_mirror_urls("https://github.com/NVIDIA/cuda-tile/archive/{}.tar.gz".format(CUDA_TILE_COMMIT)),
        series = "//third_party/cuda_tile:patches/series",
    )
