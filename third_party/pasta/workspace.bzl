"""Loads pasta python package."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "pasta",
        urls = tf_mirror_urls("https://github.com/google/pasta/archive/v0.1.8.tar.gz"),
        strip_prefix = "pasta-0.1.8",
        sha256 = "c6dc1118250487d987a7b1a404425822def2e8fb2b765eeebc96887e982b6085",
        build_file = "//third_party/pasta:pasta.BUILD",
        system_build_file = "//third_party/pasta:BUILD.system",

        # We want to add a bazel macro for use in the `@pasta` BUILD file.
        #
        # If we have this file live in this repo, referencing it from `@pasta`
        # becomes tricky. If we do `@//` the build breaks when this repo
        # (TensorFlow) is *not* the main repo (i.e. when TensorFlow is used as
        # a dependency in another workspace). If we hardcode `@org_tensorflow`,
        # the build breaks when this repo is used in another workspace under a
        # different name.
        #
        # We could generate `build_defs.bzl` to reference this repo by whatever
        # name it's registered with. Or we could just symlink `build_defs.bzl`
        # into the `@pasta` repo and then reference it with a repo relative
        # label; i.e. `//:build_defs.bzl`:
        link_files = {
            "//third_party/pasta:build_defs.bzl": "build_defs.bzl",
        },
    )
