"""Provides the repository macro to import Triton."""

load("//third_party:repo.bzl", "tf_vendored")

def repo():
    """Imports Triton."""
    tf_vendored(name = "triton", relpath = "third_party/triton")
