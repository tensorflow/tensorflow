"""Provides the repository macro to import sqlite3.

The non-standard name used aligns with TensorFlow workspace definition, to
satisfy TF presubmits.
"""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "org_sqlite",
        build_file = "//third_party/sqlite:BUILD",
        sha256 = "8a310d0a16c7a90cacd4c884e70faa51c902afed2a89f63aaa0126ab83558a32",
        strip_prefix = "sqlite-amalgamation-3530200",
        system_build_file = "//third_party/systemlibs:sqlite.BUILD",
        urls = tf_mirror_urls("https://sqlite.org/2026/sqlite-amalgamation-3530200.zip"),
    )
