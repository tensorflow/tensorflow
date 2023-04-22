"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

load("//third_party/googleapis:repository_rules.bzl", "config_googleapis")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_toolchains//repositories:repositories.bzl", bazel_toolchains_repositories = "repositories")
load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
load("@local_config_android//:android.bzl", "android_workspace")
load("@rules_cc//cc:repositories.bzl", "rules_cc_toolchains")

def _tf_bind():
    """Bind targets for some external repositories"""
    ##############################################################################
    # BIND DEFINITIONS
    #
    # Please do not add bind() definitions unless we have no other choice.
    # If that ends up being the case, please leave a comment explaining
    # why we can't depend on the canonical build target.

    # Needed by Protobuf
    native.bind(
        name = "grpc_cpp_plugin",
        actual = "@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin",
    )
    native.bind(
        name = "grpc_python_plugin",
        actual = "@com_github_grpc_grpc//src/compiler:grpc_python_plugin",
    )

    native.bind(
        name = "grpc_lib",
        actual = "@com_github_grpc_grpc//:grpc++",
    )

    native.bind(
        name = "grpc_lib_unsecure",
        actual = "@com_github_grpc_grpc//:grpc++_unsecure",
    )

    # Needed by Protobuf
    native.bind(
        name = "python_headers",
        actual = str(Label("//third_party/python_runtime:headers")),
    )

    # Needed by Protobuf
    native.bind(
        name = "six",
        actual = "@six_archive//:six",
    )

def workspace():
    http_archive(
        name = "inception_v1",
        build_file = "//:models.BUILD",
        sha256 = "7efe12a8363f09bc24d7b7a450304a15655a57a7751929b2c1593a71183bb105",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/inception_v1.zip",
        ],
    )

    http_archive(
        name = "mobile_ssd",
        build_file = "//:models.BUILD",
        sha256 = "bddd81ea5c80a97adfac1c9f770e6f55cbafd7cce4d3bbe15fbeb041e6b8f3e8",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_android_export.zip",
        ],
    )

    http_archive(
        name = "mobile_multibox",
        build_file = "//:models.BUILD",
        sha256 = "859edcddf84dddb974c36c36cfc1f74555148e9c9213dedacf1d6b613ad52b96",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/mobile_multibox_v1a.zip",
        ],
    )

    http_archive(
        name = "stylize",
        build_file = "//:models.BUILD",
        sha256 = "3d374a730aef330424a356a8d4f04d8a54277c425e274ecb7d9c83aa912c6bfa",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/stylize_v1.zip",
        ],
    )

    http_archive(
        name = "speech_commands",
        build_file = "//:models.BUILD",
        sha256 = "c3ec4fea3158eb111f1d932336351edfe8bd515bb6e87aad4f25dbad0a600d0c",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/models/speech_commands_v0.01.zip",
        ],
    )

    http_archive(
        name = "person_detect_data",
        sha256 = "170542270da256994ce24d1e357f6e84a54fdaf7d28ff2b74725a40b70b082cf",
        urls = [
            "https://storage.googleapis.com/download.tensorflow.org/data/tf_lite_micro_person_data_grayscale_2020_05_24.zip",
        ],
    )

    http_archive(
        name = "io_bazel_rules_docker",
        sha256 = "aed1c249d4ec8f703edddf35cbe9dfaca0b5f5ea6e4cd9e83e99f3b0d1136c3d",
        strip_prefix = "rules_docker-0.7.0",
        urls = [
            "https://github.com/bazelbuild/rules_docker/archive/v0.7.0.tar.gz"
        ],
    )


    rules_cc_toolchains()

    bazel_toolchains_repositories()

    # Use `swift_rules_dependencies` to fetch the toolchains. With the
    # `git_repository` rules above, the following call will skip redefining them.
    swift_rules_dependencies()

    android_workspace()

    # If a target is bound twice, the later one wins, so we have to do tf bindings
    # at the end of the WORKSPACE file.
    _tf_bind()

    grpc_extra_deps()
    config_googleapis()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_workspace0 = workspace
