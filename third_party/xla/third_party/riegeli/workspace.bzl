"""Provides the repo macro to import riegeli"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "riegeli",
        sha256 = "f63337f63f794ba9dc7dd281b20af3d036dfe0c1a5a4b7b8dc20b39f7e323b97",
        strip_prefix = "riegeli-9f2744dc23e81d84c02f6f51244e9e9bb9802d57",
        patch_file = [
            # On a version upgrade, this patch can be regenerated with the command:
            # build_tools/dependencies/gen_disable_layering_check_patch.sh.  \
            #   https://github.com/google/riegeli/archive/9f2744dc23e81d84c02f6f51244e9e9bb9802d57.tar.gz \
            #   > third_party/riegeli/layering_check.patch
            "//third_party/riegeli:layering_check.patch",
        ],
        urls = tf_mirror_urls("https://github.com/google/riegeli/archive/9f2744dc23e81d84c02f6f51244e9e9bb9802d57.tar.gz"),
    )
