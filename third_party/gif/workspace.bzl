"""Loads the gif library, used by TF."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "gif",
        build_file = "//third_party/gif:gif.BUILD",
        patch_file = [
            "//third_party/gif:gif_fix_strtok_r.patch",
            "//third_party/gif:gif_fix_image_counter.patch",
        ],
        sha256 = "31da5562f44c5f15d63340a09a4fd62b48c45620cd302f77a6d9acf0077879bd",
        strip_prefix = "giflib-5.2.1",
        system_build_file = "//third_party/systemlibs:gif.BUILD",
        urls = tf_mirror_urls("https://pilotfiber.dl.sourceforge.net/project/giflib/giflib-5.2.1.tar.gz"),
    )
