"""loads the jpeg library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "libjpeg_turbo",
        urls = tf_mirror_urls("https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.1.3.tar.gz"),
        sha256 = "dbda0c685942aa3ea908496592491e5ec8160d2cf1ec9d5fd5470e50768e7859",
        strip_prefix = "libjpeg-turbo-2.1.3",
        build_file = "//third_party/jpeg:jpeg.BUILD",
        system_build_file = "//third_party/jpeg:BUILD.system",
        patch_file = ["//third_party/jpeg:jversion.patch"],
    )
