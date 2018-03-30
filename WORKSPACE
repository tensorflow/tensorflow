workspace(name = "org_tensorflow")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "6691c58a2cd30a86776dd9bb34898b041e37136f2dc7e24cadaeaf599c95c657",
    strip_prefix = "rules_closure-08039ba8ca59f64248bb3b6ae016460fe9c9914f",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/08039ba8ca59f64248bb3b6ae016460fe9c9914f.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/08039ba8ca59f64248bb3b6ae016460fe9c9914f.tar.gz",  # 2018-01-16
    ],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

closure_repositories()

# We must check the bazel version before trying to parse any other BUILD
# files, in case the parsing of those build files depends on the bazel
# version we require here.
load("//tensorflow:version_check.bzl", "check_bazel_version_at_least")
check_bazel_version_at_least("0.10.0")

load("//tensorflow:workspace.bzl", "tf_workspace")

# Uncomment and update the paths in these entries to build the Android demo.
#android_sdk_repository(
#    name = "androidsdk",
#    api_level = 23,
#    # Ensure that you have the build_tools_version below installed in the
#    # SDK manager as it updates periodically.
#    build_tools_version = "26.0.1",
#    # Replace with path to Android SDK on your system
#    path = "<PATH_TO_SDK>",
#)
#
#android_ndk_repository(
#    name="androidndk",
#    path="<PATH_TO_NDK>",
#    # This needs to be 14 or higher to compile TensorFlow.
#    # Please specify API level to >= 21 to build for 64-bit
#    # archtectures or the Android NDK will automatically select biggest
#    # API level that it supports without notice.
#    # Note that the NDK version is not the API level.
#    api_level=14)

# Please add all new TensorFlow dependencies in workspace.bzl.
tf_workspace()

new_http_archive(
    name = "inception_v1",
    build_file = "models.BUILD",
    sha256 = "7efe12a8363f09bc24d7b7a450304a15655a57a7751929b2c1593a71183bb105",
    urls = [
        "http://storage.googleapis.com/download.tensorflow.org/models/inception_v1.zip",
        "http://download.tensorflow.org/models/inception_v1.zip",
    ],
)

new_http_archive(
    name = "mobile_ssd",
    build_file = "models.BUILD",
    sha256 = "bddd81ea5c80a97adfac1c9f770e6f55cbafd7cce4d3bbe15fbeb041e6b8f3e8",
    urls = [
        "http://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_android_export.zip",
        "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_android_export.zip",
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

new_http_archive(
    name = "speech_commands",
    build_file = "models.BUILD",
    sha256 = "c3ec4fea3158eb111f1d932336351edfe8bd515bb6e87aad4f25dbad0a600d0c",
    urls = [
        "http://storage.googleapis.com/download.tensorflow.org/models/speech_commands_v0.01.zip",
        "http://download.tensorflow.org/models/speech_commands_v0.01.zip",
    ],
)
