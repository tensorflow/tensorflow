"""Compute Library"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "compute_library",
        patch_file = [
            "//third_party/compute_library:compute_library.patch",
            "//third_party/compute_library:exclude_omp_scheduler.patch",
            "//third_party/compute_library:include_string.patch",
            "//third_party/compute_library:rules_python.patch",
        ],
        sha256 = "1bceef23aa5b3cc7321cf80e0729e87482f12544b107cdabffb88a6a52aa4adc",
        strip_prefix = "ComputeLibrary-52.8.0",
        urls = tf_mirror_urls("https://github.com/ARM-software/ComputeLibrary/archive/refs/tags/v52.8.0.tar.gz"),
    )
