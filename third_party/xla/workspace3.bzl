"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# buildifier: disable=function-docstring
# buildifier: disable=unnamed-macro
def workspace():
    tf_http_archive(
        name = "io_bazel_rules_closure",
        sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
        strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
        ),
    )

    # https://github.com/bazelbuild/bazel-skylib/releases
    tf_http_archive(
        name = "bazel_skylib",
        sha256 = "bc283cdfcd526a52c3201279cda4bc298652efa898b10b4db0837dc51652756f",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.7.1/bazel-skylib-1.7.1.tar.gz",
        ),
    )

    tf_http_archive(
        name = "rules_license",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/rules_license/releases/download/0.0.7/rules_license-0.0.7.tar.gz",
        ),
        sha256 = "4531deccb913639c30e5c7512a054d5d875698daeb75d8cf90f284375fe7c360",
    )

    tf_http_archive(
        name = "rules_pkg",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.7.1/rules_pkg-0.7.1.tar.gz",
        ),
        sha256 = "451e08a4d78988c06fa3f9306ec813b836b1d076d0f055595444ba4ff22b867f",
    )

    tf_http_archive(
        name = "bazel_features",
        sha256 = "4fd9922d464686820ffd8fcefa28ccffa147f7cdc6b6ac0d8b07fde565c65d66",
        strip_prefix = "bazel_features-1.25.0",
        urls = tf_mirror_urls("https://github.com/bazel-contrib/bazel_features/releases/download/v1.25.0/bazel_features-v1.25.0.tar.gz"),
    )

    # Toolchains for ML projects hermetic builds.
    # Details: https://github.com/google-ml-infra/rules_ml_toolchain
    tf_http_archive(
        name = "rules_ml_toolchain",
        sha256 = "40963e4bc262dfa9a43146f610140af0068b023ace8f3c50f1705a7b50de0830",
        strip_prefix = "rules_ml_toolchain-cad1047facbac4fb3c1124da68bf2cb36c7eb9ac",
        urls = tf_mirror_urls(
            "https://github.com/google-ml-infra/rules_ml_toolchain/archive/cad1047facbac4fb3c1124da68bf2cb36c7eb9ac.tar.gz",
        ),
    )

    # Maven dependencies.
    RULES_JVM_EXTERNAL_TAG = "4.3"
    tf_http_archive(
        name = "rules_jvm_external",
        strip_prefix = "rules_jvm_external-%s" % RULES_JVM_EXTERNAL_TAG,
        sha256 = "6274687f6fc5783b589f56a2f1ed60de3ce1f99bc4e8f9edef3de43bdf7c6e74",
        urls = tf_mirror_urls("https://github.com/bazelbuild/rules_jvm_external/archive/%s.zip" % RULES_JVM_EXTERNAL_TAG),
    )

    # Platforms
    tf_http_archive(
        name = "platforms",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/platforms/releases/download/0.0.11/platforms-0.0.11.tar.gz",
        ),
        sha256 = "29742e87275809b5e598dc2f04d86960cc7a55b3067d97221c9abbc9926bff0f",
    )

    # clang-tidy checks.
    tf_http_archive(
        name = "bazel_clang_tidy",
        strip_prefix = "bazel_clang_tidy-c4d35e0d0b838309358e57a2efed831780f85cd0",
        sha256 = "96da6e935ccc91045cf928dbc57f22508a2729c51f7fb3f56178017b0deb9b3c",
        urls = tf_mirror_urls("https://github.com/erenon/bazel_clang_tidy/archive/c4d35e0d0b838309358e57a2efed831780f85cd0.tar.gz"),
    )

    tf_http_archive(
        name = "tar.bzl",
        sha256 = "2799dfd43131e872c8f9fed41a547a8c5afd571c5df6b809eab6219d44403872",
        strip_prefix = "tar.bzl-0.10.5",
        urls = tf_mirror_urls("https://github.com/bazel-contrib/tar.bzl/releases/download/v0.10.5/tar.bzl-v0.10.5.tar.gz"),
    )

    tf_http_archive(
        name = "jq.bzl",
        sha256 = "7b63435aa19cc6a0cfd1a82fbdf2c7a2f0a94db1a79ff7a4469ffa94286261ab",
        strip_prefix = "jq.bzl-0.1.0",
        urls = tf_mirror_urls("https://github.com/bazel-contrib/jq.bzl/releases/download/v0.1.0/jq.bzl-v0.1.0.tar.gz"),
    )

    tf_http_archive(
        name = "yq.bzl",
        sha256 = "b51d82b561a78ab21d265107b0edbf98d68a390b4103992d0b03258bb3819601",
        strip_prefix = "yq.bzl-0.1.1",
        urls = tf_mirror_urls("https://github.com/bazel-contrib/yq.bzl/releases/download/v0.1.1/yq.bzl-v0.1.1.tar.gz"),
    )

    tf_http_archive(
        name = "aspect_bazel_lib",
        sha256 = "db7da732db4dece80cd6d368220930950c9306ff356ebba46498fe64e65a3945",
        strip_prefix = "bazel-lib-2.19.3",
        urls = tf_mirror_urls("https://github.com/aspect-build/bazel-lib/releases/download/v2.19.3/bazel-lib-v2.19.3.tar.gz"),
    )

    tf_http_archive(
        name = "bazel_lib",
        sha256 = "db7da732db4dece80cd6d368220930950c9306ff356ebba46498fe64e65a3945",
        strip_prefix = "bazel-lib-2.19.3",
        urls = tf_mirror_urls("https://github.com/aspect-build/bazel-lib/releases/download/v2.19.3/bazel-lib-v2.19.3.tar.gz"),
    )

    tf_http_archive(
        name = "aspect_rules_js",
        sha256 = "9dd50d3bacb2fe1d4a721098981b70290fe9ac56d3625791f490d2ab94f2cac6",
        strip_prefix = "rules_js-2.7.0",
        urls = tf_mirror_urls("https://github.com/aspect-build/rules_js/releases/download/v2.7.0/rules_js-v2.7.0.tar.gz"),
    )

    tf_http_archive(
        name = "aspect_rules_esbuild",
        sha256 = "550e33ddeb86a564b22b2c5d3f84748c6639b1b2b71fae66bf362c33392cbed8",
        strip_prefix = "rules_esbuild-0.21.0",
        urls = tf_mirror_urls("https://github.com/aspect-build/rules_esbuild/releases/download/v0.21.0/rules_esbuild-v0.21.0.tar.gz"),
    )

    tf_http_archive(
        name = "aspect_rules_ts",
        sha256 = "6fd16aa24c2e8547b72561ece1c7d307b77a5f98f0402934396f6eefbac59aa2",
        strip_prefix = "rules_ts-3.7.0",
        urls = tf_mirror_urls("https://github.com/aspect-build/rules_ts/releases/download/v3.7.0/rules_ts-v3.7.0.tar.gz"),
    )

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
xla_workspace3 = workspace
