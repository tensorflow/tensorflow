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
        sha256 = "458b658277ba51b4730ea7a2020efdf1c6dcadf7d30de72e37f4308277fa8c01",
        strip_prefix = "rules_cc-0.2.16",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_cc/releases/download/0.2.16/rules_cc-0.2.16.tar.gz"),
    )

    tf_http_archive(
        name = "com_google_protobuf",
        patch_file = ["@xla//third_party/protobuf:protobuf.patch"],
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
        sha256 = "fa7dd2c6b7d63b3585028dd8a90a6cf9db83c33b250959c2ee7b583a6c130e12",
        strip_prefix = "rules_python-1.6.0",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_python/releases/download/1.6.0/rules_python-1.6.0.tar.gz"),
        patch_file = [
            "@xla//third_party/py:rules_python_pip_version.patch",
            "@xla//third_party/py:rules_python_freethreaded.patch",
            "@xla//third_party/py:rules_python_versions.patch",
        ] + extra_patches,
    )
