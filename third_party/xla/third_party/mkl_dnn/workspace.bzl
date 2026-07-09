"""oneAPI Deep Neural Network Library (oneDNN)"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "onednn",
        build_file = "//third_party/mkl_dnn:mkldnn_v1.BUILD",
        patch_file = ["//third_party/mkl_dnn:setting_init.patch"],
        sha256 = "071f289dc961b43a3b7c8cbe8a305290a7c5d308ec4b2f586397749abdc88296",
        strip_prefix = "oneDNN-3.7.3",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.7.3.tar.gz"),
    )

    tf_http_archive(
        name = "onednn_async",
        build_file = "//third_party/mkl_dnn:mkldnn_v1_async.BUILD",
        # Rename the gtest module in oneDNN's third_party to avoid
        # conflict with Google's gtest module
        patch_file = [
            "//third_party/mkl_dnn:setting_init.patch",
            "//third_party/mkl_dnn:rename_gtest.patch",
        ],
        sha256 = "7293a85e146c2710dcf4f7257fdebb91020004cf1627c8de684b814c2498c81a",
        strip_prefix = "oneDNN-3.11.3",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.11.3.tar.gz"),
    )

    tf_http_archive(
        name = "mkl_dnn_acl_compatible",
        build_file = "//third_party/mkl_dnn:mkldnn_acl.BUILD",
        patch_file = [
            "//third_party/mkl_dnn:setting_init.patch",
            "//third_party/mkl_dnn:rename_gtest.patch",
            "//third_party/mkl_dnn:remove_unused_ranges_include.patch",
            "//third_party/mkl_dnn:fix_acl_benchmark_scheduler_name.patch",
        ],
        sha256 = "7293a85e146c2710dcf4f7257fdebb91020004cf1627c8de684b814c2498c81a",
        strip_prefix = "oneDNN-3.11.3",
        urls = tf_mirror_urls("https://github.com/oneapi-src/oneDNN/archive/refs/tags/v3.11.3.tar.gz"),
    )
