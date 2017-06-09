workspace(name = "org_tensorflow")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "bc41b80486413aaa551860fc37471dbc0666e1dbb5236fb6177cb83b0c105846",
    strip_prefix = "rules_closure-dec425a4ff3faf09a56c85d082e4eed05d8ce38f",
    urls = [
        "http://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dec425a4ff3faf09a56c85d082e4eed05d8ce38f.tar.gz",  # 2017-06-02
        "https://github.com/bazelbuild/rules_closure/archive/dec425a4ff3faf09a56c85d082e4eed05d8ce38f.tar.gz",
    ],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

closure_repositories()

load("//tensorflow:workspace.bzl", "tf_workspace")

# Uncomment and update the paths in these entries to build the Android demo.
#android_sdk_repository(
#    name = "androidsdk",
#    api_level = 23,
#    # Ensure that you have the build_tools_version below installed in the
#    # SDK manager as it updates periodically.
#    build_tools_version = "25.0.2",
#    # Replace with path to Android SDK on your system
#    path = "<PATH_TO_SDK>",
#)
#
# Android NDK r12b is recommended (higher may cause issues with Bazel)
#android_ndk_repository(
#    name="androidndk",
#    path="<PATH_TO_NDK>",
#    # This needs to be 14 or higher to compile TensorFlow.
#    # Note that the NDK version is not the API level.
#    api_level=14)

# Please add all new TensorFlow dependencies in workspace.bzl.
tf_workspace()

new_http_archive(
    name = "inception5h",
    build_file = "models.BUILD",
    sha256 = "d13569f6a98159de37e92e9c8ec4dae8f674fbf475f69fe6199b514f756d4364",
    urls = [
        "http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip",
        "http://download.tensorflow.org/models/inception5h.zip",
    ],
)

new_http_archive(
    name = "mobile_multibox",
    build_file = "models.BUILD",
    sha256 = "859edcddf84dddb974c36c36cfc1f74555148e9c9213dedacf1d6b613ad52b96",
    urls = [
        "http://storage.googleapis.com/download.tensorflow.org/models/mobile_multibox_v1a.zip",
        "http://download.tensorflow.org/models/mobile_multibox_v1a.zip",
    ],
)

new_http_archive(
    name = "stylize",
    build_file = "models.BUILD",
    sha256 = "3d374a730aef330424a356a8d4f04d8a54277c425e274ecb7d9c83aa912c6bfa",
    urls = [
        "http://storage.googleapis.com/download.tensorflow.org/models/stylize_v1.zip",
        "http://download.tensorflow.org/models/stylize_v1.zip",
    ],
)
