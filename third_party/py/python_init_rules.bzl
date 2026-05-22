"""Hermetic Python initialization. Consult the WORKSPACE on how to use it."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def python_init_rules(extra_patches = []):
    """Defines (doesn't setup) the rules_python repository.

    Args:
      extra_patches: list of labels. Additional patches to apply after the default
        set of patches.
    """

    http_archive(
        name = "rules_cc",
        urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.2.11/rules_cc-0.2.11.tar.gz"],
        strip_prefix = "rules_cc-0.2.11",
        sha256 = "5287821524d1c1d20f1c0ffa90bd2c2d776473dd8c84dafa9eb783150286d825",
    )

    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = [
            "@xla//third_party/protobuf:protobuf.patch",
            "@xla//third_party/protobuf:protobuf_arena.patch",
        ],
        sha256 = "6e09bbc950ba60c3a7b30280210cd285af8d7d8ed5e0a6ed101c72aff22e8d88",
        strip_prefix = "protobuf-6.31.1",
        urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/archive/refs/tags/v6.31.1.zip"),
        repo_mapping = {
            "@abseil-cpp": "@com_google_absl",
            "@protobuf_pip_deps": "@pypi",
        },
    )

    tf_http_archive(
        name = "rules_python",
        sha256 = "8964aa1e7525fea5244ba737458694a057ada1be96a92998a41caa1983562d00",
        strip_prefix = "rules_python-1.8.5",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_python/releases/download/1.8.5/rules_python-1.8.5.tar.gz"),
        patch_file = [
            "@xla//third_party/py:rules_python_pip_version.patch",
            "@xla//third_party/py:rules_python_scope.patch",
            "@xla//third_party/py:rules_python_freethreaded.patch",
            "@xla//third_party/py:rules_python_versions.patch",
        ] + extra_patches,
    )
