"""Provides the repository macro to import Tensor IR."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    """Imports Tensor IR."""
    TENSOR_IR_COMMIT = "63692d79629e6f32a1d8757695590a59e0adbafd"
    TENSOR_IR_SHA256 = "b80794d7c2bfb1bc1ca432d892977becba8d35ec6c18c586acbd648ccc8074dd"

    tf_http_archive(
        name = "tensor_ir",
        build_file = "//third_party/tensor_ir:tensor_ir.BUILD",
        sha256 = TENSOR_IR_SHA256,
        strip_prefix = "tensor-ir-{}".format(TENSOR_IR_COMMIT),
        urls = tf_mirror_urls("https://github.com/NVIDIA/tensor-ir/archive/{}.tar.gz".format(TENSOR_IR_COMMIT)),
        patch_file = [
            "//third_party/tensor_ir:patches/unused_variable.patch",
            "//third_party/tensor_ir:patches/symbol_op_interface.patch",
        ],
    )
