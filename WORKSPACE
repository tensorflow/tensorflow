workspace(name = "org_tensorflow")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "a38539c5b5c358548e75b44141b4ab637bba7c4dc02b46b1f62a96d6433f56ae",
    strip_prefix = "rules_closure-dbb96841cc0a5fb2664c37822803b06dab20c7d1",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/dbb96841cc0a5fb2664c37822803b06dab20c7d1.tar.gz",  # 2018-04-13
    ],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

closure_repositories()

# We must check the bazel version before trying to parse any other BUILD
# files, in case the parsing of those build files depends on the bazel
# version we require here.
load("//tensorflow:version_check.bzl", "check_bazel_version_at_least")
check_bazel_version_at_least("0.15.0")

load("//tensorflow:workspace.bzl", "tf_workspace")

load("//third_party/android:android_configure.bzl", "android_configure")
android_configure(name="local_config_android")
load("@local_config_android//:android.bzl", "android_workspace")
android_workspace()

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
