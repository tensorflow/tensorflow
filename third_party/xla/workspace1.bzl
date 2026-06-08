"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load(
    "@aspect_bazel_lib//lib:repositories.bzl",
    "register_copy_directory_toolchains",
    "register_copy_to_directory_toolchains",
    "register_coreutils_toolchains",
    "register_expand_template_toolchains",
    "register_jq_toolchains",
    "register_tar_toolchains",
    "register_yq_toolchains",
)
load("@aspect_rules_esbuild//esbuild:repositories.bzl", "LATEST_ESBUILD_VERSION", "esbuild_register_toolchains")
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
load("@rules_cc//cc:extensions.bzl", "compatibility_proxy_repo")
load("@rules_nodejs//nodejs:repositories.bzl", "DEFAULT_NODE_VERSION", "nodejs_register_toolchains")
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/llvm:setup.bzl", "llvm_setup")

# buildifier: disable=unnamed-macro
def workspace():
    """Loads a set of TensorFlow dependencies in a WORKSPACE file."""
    llvm_setup(name = "llvm-project")
    native.register_toolchains("@local_config_python//:py_toolchain")
    rules_pkg_dependencies()
    nodejs_register_toolchains(
        name = "nodejs",
        node_version = DEFAULT_NODE_VERSION,
    )
    register_copy_directory_toolchains()
    register_copy_to_directory_toolchains()
    register_coreutils_toolchains()
    register_expand_template_toolchains()
    register_jq_toolchains()
    register_tar_toolchains()
    register_yq_toolchains()
    esbuild_register_toolchains(
        name = "esbuild",
        esbuild_version = LATEST_ESBUILD_VERSION,
    )
    compatibility_proxy_repo()

    tf_http_archive(
        name = "bazel_toolchains",
        sha256 = "294cdd859e57fcaf101d4301978c408c88683fbc46fbc1a3829da92afbea55fb",
        strip_prefix = "bazel-toolchains-8c717f8258cd5f6c7a45b97d974292755852b658",
        urls = tf_mirror_urls(
            "https://github.com/bazelbuild/bazel-toolchains/archive/8c717f8258cd5f6c7a45b97d974292755852b658.tar.gz",
        ),
    )

    grpc_deps()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
xla_workspace1 = workspace
