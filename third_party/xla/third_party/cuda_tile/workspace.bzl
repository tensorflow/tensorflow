"""Provides the repository macro to import CUDA Tile IR."""

load("//third_party:quilt_repo.bzl", "quilt_http_archive")
load("//third_party:repo.bzl", "tf_mirror_urls")

def repo():
    """Imports CUDA Tile IR."""
    CUDA_TILE_COMMIT = "e11729ed4e228af005dc8cc113d7b84157e9d654"
    CUDA_TILE_SHA256 = "56c03d732ac5f23e0ae9c0c184ff65aa282c24bd30a1bde723457f8e986e855e"

    quilt_http_archive(
        name = "cuda_tile",
        build_file = "//third_party/cuda_tile:cuda_tile.BUILD",
        sha256 = CUDA_TILE_SHA256,
        strip_prefix = "cuda-tile-{}".format(CUDA_TILE_COMMIT),
        urls = tf_mirror_urls("https://github.com/NVIDIA/cuda-tile/archive/{}.tar.gz".format(CUDA_TILE_COMMIT)),
        series = "//third_party/cuda_tile:patches/series",
    )
