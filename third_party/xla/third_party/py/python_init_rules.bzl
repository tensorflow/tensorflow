"""Hermetic Python initialization. Consult the WORKSPACE on how to use it."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def python_init_rules():
    http_archive(
        name = "rules_python",
        sha256 = "62ddebb766b4d6ddf1712f753dac5740bea072646f630eb9982caa09ad8a7687",
        strip_prefix = "rules_python-0.39.0",
        url = "https://github.com/bazelbuild/rules_python/releases/download/0.39.0/rules_python-0.39.0.tar.gz",
        patch_args = ["-p1"],
        patches = [
            Label("//third_party/py:rules_python1.patch"),
            Label("//third_party/py:rules_python2.patch"),
            Label("//third_party/py:rules_python3.patch"),
        ],
    )
