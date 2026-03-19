"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("@bazel_toolchains//repositories:repositories.bzl", bazel_toolchains_repositories = "repositories")
load("@build_bazel_apple_support//lib:repositories.bzl", "apple_support_dependencies")
load("@build_bazel_rules_apple//apple:repositories.bzl", "apple_rules_dependencies")
load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
load("@local_config_android//:android.bzl", "android_workspace")
load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
load("//third_party:models_repos.bzl", "models_repositories")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/googleapis:repository_rules.bzl", "config_googleapis")

def _tf_bind():
    """Bind targets for some external repositories"""
    ##############################################################################
    # BIND DEFINITIONS
    #
    # Please do not add bind() definitions unless we have no other choice.
    # If that ends up being the case, please leave a comment explaining
    # why we can't depend on the canonical build target.

def workspace():
    """TensorFlow workspace initialization."""
    models_repositories()
    bazel_toolchains_repositories()

    # Apple rules for Bazel. https://github.com/bazelbuild/rules_apple.
    # Note: We add this to fix Kokoro builds.
    # The rules below call into `rules_proto` but the hash has changed and
    # Bazel refuses to continue. So, we add our own mirror.
    tf_http_archive(
        name = "rules_proto",
        sha256 = "20b240eba17a36be4b0b22635aca63053913d5c1ee36e16be36499d167a2f533",
        strip_prefix = "rules_proto-11bf7c25e666dd7ddacbcd4d4c4a9de7a25175f8",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/rules_proto/archive/11bf7c25e666dd7ddacbcd4d4c4a9de7a25175f8.tar.gz",
        ),
    )

    # Now, finally use the rules
    apple_rules_dependencies()
    swift_rules_dependencies()
    apple_support_dependencies()

    android_workspace()

    # If a target is bound twice, the later one wins, so we have to do tf bindings
    # at the end of the WORKSPACE file.
    _tf_bind()

    grpc_extra_deps()
    rules_foreign_cc_dependencies()
    config_googleapis()

    # Toolchains for ML projects hermetic builds.
    # Details: https://github.com/google-ml-infra/rules_ml_toolchain
    tf_http_archive(
        name = "rules_ml_toolchain",
        sha256 = "54c1a357f71f611efdb4891ebd4bcbe4aeb6dfa7e473f14fd7ecad5062096616",
        strip_prefix = "rules_ml_toolchain-d8cb9c2c168cd64000eaa6eda0781a9615a26ffe",
        urls = tf_mirror_urls(
            "https://github.com/google-ml-infra/rules_ml_toolchain/archive/d8cb9c2c168cd64000eaa6eda0781a9615a26ffe.tar.gz",
        ),
    )

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_workspace0 = workspace
