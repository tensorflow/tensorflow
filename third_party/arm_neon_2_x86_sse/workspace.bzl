"""Loads the arm_neon_2_x86_sse library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "arm_neon_2_x86_sse",
        build_file = "//third_party/arm_neon_2_x86_sse:arm_neon_2_x86_sse.BUILD",
        sha256 = "019fbc7ec25860070a1d90e12686fc160cfb33e22aa063c80f52b363f1361e9d",
        strip_prefix = "ARM_NEON_2_x86_SSE-a15b489e1222b2087007546b4912e21293ea86ff",
        urls = tf_mirror_urls("https://github.com/intel/ARM_NEON_2_x86_SSE/archive/a15b489e1222b2087007546b4912e21293ea86ff.tar.gz"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/neon2sse.cmake)
