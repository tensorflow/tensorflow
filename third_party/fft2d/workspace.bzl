"""FFT2D"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    # LINT.IfChange
    tf_http_archive(
        name = "fft2d",
        build_file = "//third_party/fft2d:fft2d.BUILD",
        sha256 = "5f4dabc2ae21e1f537425d58a49cdca1c49ea11db0d6271e2a4b27e9697548eb",
        strip_prefix = "OouraFFT-1.0",
        urls = tf_mirror_urls("https://github.com/petewarden/OouraFFT/archive/v1.0.tar.gz"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/fft2d.cmake)
