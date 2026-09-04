"""Provides the repository macro to import CUDA Tile IR."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports CUDA Tile IR."""
    CUDA_TILE_COMMIT = "af2417041cc939b87ef56d92cfdcf61737c5457e"
    CUDA_TILE_SHA256 = "81597e49469171bf8fa7319fbd44ebe133001521f484589e3dd3fb3fad282dc0"

    tf_http_archive(
        name = "cuda_tile",
        build_file = "//third_party/cuda_tile:cuda_tile.BUILD",
        sha256 = CUDA_TILE_SHA256,
        strip_prefix = "cuda-tile-{}".format(CUDA_TILE_COMMIT),
        urls = tf_mirror_urls("https://github.com/NVIDIA/cuda-tile/archive/{}.tar.gz".format(CUDA_TILE_COMMIT)),
        patch_file = [
            "//third_party/cuda_tile:patches/constructor.patch",
            "//third_party/cuda_tile:patches/symbol_op_interface.patch",
        ],
    )
