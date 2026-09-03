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
            "@xla//third_party/protobuf:protobuf_arena.patch",
            "@xla//third_party/protobuf:fix_message_lite_incomplete_type.patch",
            "@xla//third_party/protobuf:fix_python_dist_package.patch",
            "@xla//third_party/protobuf:nodiscard.patch",
        ],
        sha256 = "61e5e5b7f29c4a719d9691b97c2b8937b8bd5ab1b6b7586f3f55934011806280",
        strip_prefix = "protobuf-34.1",
        urls = tf_mirror_urls("https://github.com/protocolbuffers/protobuf/releases/download/v34.1/protobuf-34.1.zip"),
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
        ] + [p for p in extra_patches if "site_init_retry" not in str(p)],
    )
