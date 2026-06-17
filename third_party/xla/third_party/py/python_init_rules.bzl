"""Hermetic Python initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def python_init_rules(extra_patches = []):
    """Defines (doesn't setup) the rules_python repository.

    Args:
      extra_patches: list of labels. Additional patches to apply after the default
        set of patches.
    """

    tf_http_archive(
        name = "rules_cc",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_cc/archive/refs/tags/0.1.0.tar.gz"),
        strip_prefix = "rules_cc-0.1.0",
        sha256 = "4b12149a041ddfb8306a8fd0e904e39d673552ce82e4296e96fac9cbf0780e59",
        patch_file = [
            "@xla//third_party/py:rules_cc_protobuf.patch",
        ],
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
