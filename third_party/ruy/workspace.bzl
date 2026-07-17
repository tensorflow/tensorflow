"""Loads the ruy library, used by TensorFlow Lite."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "ruy",
        # LINT.IfChange
        sha256 = "64a001a81c05743736ed8695edd521c6093519ef2edb6635ffa142e827cfb86b",
        strip_prefix = "ruy-2af88863614a8298689cc52b1a47b3fcad7be835",
        urls = tf_mirror_urls("https://github.com/google/ruy/archive/2af88863614a8298689cc52b1a47b3fcad7be835.zip"),
        # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/ruy.cmake)
        build_file = "//third_party/ruy:BUILD",
        # Override Google-internal BUILD files and .bzl files with OSS-compatible versions.
        # The new ruy commit contains Google-internal Bazel paths that are invalid in
        # open-source builds (internal rules_cc loads, internal windows conditions, etc).
        link_files = {
            "//third_party/ruy:ruy_BUILD": "ruy/BUILD.bazel",
            "//third_party/ruy:ruy_profiler_BUILD": "ruy/profiler/BUILD.bazel",
            "//third_party/ruy:ruy_test.oss.bzl": "ruy/ruy_test.bzl",
            "//third_party/ruy:build_defs.oss_override.bzl": "ruy/build_defs.bzl",
            "//third_party/ruy:build_defs_oss.oss_override.bzl": "ruy/build_defs.oss.bzl",
        },
    )
