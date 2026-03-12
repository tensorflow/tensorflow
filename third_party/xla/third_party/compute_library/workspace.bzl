"""Compute Library"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "compute_library",
        patch_file = [
            "//third_party/compute_library:acl_gemm_scheduling_heuristic.patch",
            "//third_party/compute_library:acl_stateless_gemm_workspace.patch",
            "//third_party/compute_library:compute_library.patch",
            "//third_party/compute_library:exclude_omp_scheduler.patch",
            "//third_party/compute_library:include_string.patch",
        ],
        sha256 = "8273f68cd0bb17e9231a11a6618d245eb6d623884ae681c00e7a4eabca2dad42",
        strip_prefix = "ComputeLibrary-24.12",
        urls = tf_mirror_urls("https://github.com/ARM-software/ComputeLibrary/archive/refs/tags/v24.12.tar.gz"),
    )
