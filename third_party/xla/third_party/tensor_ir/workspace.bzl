"""Provides the repository macro to import Tensor IR."""

load("//third_party:quilt_repo.bzl", "quilt_http_archive")
load("//third_party:repo.bzl", "tf_mirror_urls")

def repo():
    """Imports Tensor IR."""
    TENSOR_IR_COMMIT = "f6e7e8c6270b119cdcbcfc5e952cf9186bdf074e"
    TENSOR_IR_SHA256 = "448bf4dbe233605c726c8cb2be71228aa9e20796e5c2c72d644756bb501ab6c4"

    quilt_http_archive(
        name = "tensor_ir",
        build_file = "//third_party/tensor_ir:tensor_ir.BUILD",
        sha256 = TENSOR_IR_SHA256,
        strip_prefix = "tensor-ir-{}".format(TENSOR_IR_COMMIT),
        urls = tf_mirror_urls("https://github.com/NVIDIA/tensor-ir/archive/{}.tar.gz".format(TENSOR_IR_COMMIT)),
        series = "//third_party/tensor_ir:patches/series",
    )
