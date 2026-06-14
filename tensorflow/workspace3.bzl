"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls", "tf_vendored")
load("//third_party/tf_runtime:workspace.bzl", tf_runtime = "repo")

def _cc_compatibility_proxy_impl(repository_ctx):
    repository_ctx.file("BUILD", "exports_files(['proxy.bzl', 'symbols.bzl'])")
    repository_ctx.file("proxy.bzl", """
cc_binary = native.cc_binary
cc_library = native.cc_library
cc_import = native.cc_import
cc_test = native.cc_test
cc_proto_library = native.cc_proto_library
cc_shared_library = native.cc_shared_library
objc_import = native.objc_import
objc_library = native.objc_library
""")
    repository_ctx.file("symbols.bzl", """
cc_common = native.cc_common
""")

cc_compatibility_proxy = repository_rule(
    implementation = _cc_compatibility_proxy_impl,
)

def workspace():
    # Define rules_cc 0.2.11
    http_archive(
        name = "rules_cc",
        urls = ["https://github.com/bazelbuild/rules_cc/releases/download/0.2.11/rules_cc-0.2.11.tar.gz"],
        strip_prefix = "rules_cc-0.2.11",
        sha256 = "5287821524d1c1d20f1c0ffa90bd2c2d776473dd8c84dafa9eb783150286d825",
    )

    # Initialize custom compatibility proxy early
    cc_compatibility_proxy(name = "cc_compatibility_proxy")

    tf_vendored(name = "xla", path = "third_party/xla")
    tf_vendored(name = "tsl", path = "third_party/xla/third_party/tsl")

    tf_http_archive(
        name = "io_bazel_rules_closure",
        sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
        strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
        ),
    )

    tf_runtime()

    # https://github.com/bazelbuild/bazel-skylib/releases
    http_archive(
        name = "bazel_skylib",
        sha256 = "6e78f0e57de26801f6f564fa7c4a48dc8b36873e416257a92bbb0937eeac8446",
        urls = [
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.8.2/bazel-skylib-1.8.2.tar.gz",
        ],
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
        urls = tf_mirror_urls(
            "https://github.com/bazel-contrib/bazel_features/releases/download/v1.25.0/bazel_features-v1.25.0.tar.gz",
        ),
    )

    # Toolchains for ML projects hermetic builds.
    # Details: https://github.com/google-ml-infra/rules_ml_toolchain
    tf_http_archive(
        name = "rules_ml_toolchain",
        sha256 = "0b42f693a60c6050d87db1e0a0eaeb84ab3f54191fce094d86334faedc807da0",
        strip_prefix = "rules_ml_toolchain-398d613aea7a4c294da49b79a6d6f3f8732bd84c",
        urls = tf_mirror_urls(
            "https://github.com/google-ml-infra/rules_ml_toolchain/archive/398d613aea7a4c294da49b79a6d6f3f8732bd84c.tar.gz",
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
        sha256 = "29742e87275809b5e598dc2f04d86960cc7a55b3067d97221c9abbc9926bff0f",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/platforms/releases/download/0.0.11/platforms-0.0.11.tar.gz",
        ),
    )

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_workspace3 = workspace
