"""Hermetic Python initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def python_init_rules(extra_patches = []):
    """Defines (doesn't setup) the rules_python repository.

    Args:
      extra_patches: list of labels. Additional patches to apply after the default
        set of patches.
    """

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
        sha256 = "e11d2e1efce1589e5bdfa93986712c74fc7467a0f093143d489d2ef5ebb1ed2a",
        strip_prefix = "rules_python-2.2.0",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_python/releases/download/2.2.0/rules_python-2.2.0.tar.gz"),
        # TODO(phawkins): Remove this filter once JAX removes rules_python_site_init_retry.patch.
        # rules_python 2.2.0 has the Windows retry fix upstreamed.
        patch_file = [
            "@xla//third_party/py:rules_python_scope.patch",
            "@xla//third_party/py:rules_python_freethreaded.patch",
            "@xla//third_party/py:rules_python_versions.patch",
            "@xla//third_party/py:rules_python_windows_aarch64.patch",
        ] + [p for p in extra_patches if "site_init_retry" not in str(p)],
    )
