workspace(name = "org_tensorflow")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "e0a111000aeed2051f29fcc7a3f83be3ad8c6c93c186e64beb1ad313f0c7f9f9",
    strip_prefix = "rules_closure-cf1e44edb908e9616030cc83d085989b8e6cd6df",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",  # 2019-04-04
    ],
)

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")

closure_repositories()

load("//third_party/toolchains/preconfig/generate:archives.bzl",
     "bazel_toolchains_archive")

bazel_toolchains_archive()

load(
    "@bazel_toolchains//repositories:repositories.bzl",
    bazel_toolchains_repositories = "repositories",
)

bazel_toolchains_repositories()

load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)

container_repositories()

load("//third_party/toolchains/preconfig/generate:workspace.bzl",
     "remote_config_workspace")

remote_config_workspace()

# Apple and Swift rules.
http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "8f32e2839fba28d549e1670dbed83606dd339a9f7489118e481814d61738270f",
    urls = ["https://github.com/bazelbuild/rules_apple/releases/download/0.14.0/rules_apple.0.14.0.tar.gz"],
)  # https://github.com/bazelbuild/rules_apple/releases
http_archive(
    name = "build_bazel_apple_support",
    sha256 = "7356dbd44dea71570a929d1d4731e870622151a5f27164d966dda97305f33471",
    urls = ["https://github.com/bazelbuild/apple_support/releases/download/0.6.0/apple_support.0.6.0.tar.gz"],
)  # https://github.com/bazelbuild/apple_support/releases
http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz"],
)  # https://github.com/bazelbuild/bazel-skylib/releases
http_archive(
    name = "build_bazel_rules_swift",
    sha256 = "31aad005a9c4e56b256125844ad05eb27c88303502d74138186f9083479f93a6",
    urls = ["https://github.com/bazelbuild/rules_swift/releases/download/0.8.0/rules_swift.0.8.0.tar.gz"],
)  # https://github.com/bazelbuild/rules_swift/releases
http_archive(
    name = "com_github_apple_swift_swift_protobuf",
    type = "zip",
    strip_prefix = "swift-protobuf-1.4.0/",
    urls = ["https://github.com/apple/swift-protobuf/archive/1.4.0.zip"],
)  # https://github.com/apple/swift-protobuf/releases
http_file(
    name = "xctestrunner",
    executable = 1,
    urls = ["https://github.com/google/xctestrunner/releases/download/0.2.7/ios_test_runner.par"],
)  # https://github.com/google/xctestrunner/releases
# Use `swift_rules_dependencies` to fetch the toolchains. With the
# `git_repository` rules above, the following call will skip redefining them.
load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")
swift_rules_dependencies()

# We must check the bazel version before trying to parse any other BUILD
# files, in case the parsing of those build files depends on the bazel
# version we require here.
load("//tensorflow:version_check.bzl", "check_bazel_version_at_least")
check_bazel_version_at_least("0.19.0")

load("//tensorflow:workspace.bzl", "tf_workspace")

load("//third_party/android:android_configure.bzl", "android_configure")
android_configure(name="local_config_android")
load("@local_config_android//:android.bzl", "android_workspace")
android_workspace()

# Please add all new TensorFlow dependencies in workspace.bzl.
tf_workspace()

http_archive(
    name = "inception_v1",
    build_file = "//:models.BUILD",
    sha256 = "7efe12a8363f09bc24d7b7a450304a15655a57a7751929b2c1593a71183bb105",
    urls = [
        "http://storage.googleapis.com/download.tensorflow.org/models/inception_v1.zip",
        "http://download.tensorflow.org/models/inception_v1.zip",
    ],
)

http_archive(
    name = "mobile_ssd",
    build_file = "//:models.BUILD",
    sha256 = "bddd81ea5c80a97adfac1c9f770e6f55cbafd7cce4d3bbe15fbeb041e6b8f3e8",
    urls = [
        "http://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_android_export.zip",
        "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_android_export.zip",
    ],
)

http_archive(
    name = "mobile_multibox",
    build_file = "//:models.BUILD",
    sha256 = "859edcddf84dddb974c36c36cfc1f74555148e9c9213dedacf1d6b613ad52b96",
    urls = [
        "http://storage.googleapis.com/download.tensorflow.org/models/mobile_multibox_v1a.zip",
        "http://download.tensorflow.org/models/mobile_multibox_v1a.zip",
    ],
)

http_archive(
    name = "stylize",
    build_file = "//:models.BUILD",
    sha256 = "3d374a730aef330424a356a8d4f04d8a54277c425e274ecb7d9c83aa912c6bfa",
    urls = [
        "http://storage.googleapis.com/download.tensorflow.org/models/stylize_v1.zip",
        "http://download.tensorflow.org/models/stylize_v1.zip",
    ],
)

http_archive(
    name = "speech_commands",
    build_file = "//:models.BUILD",
    sha256 = "c3ec4fea3158eb111f1d932336351edfe8bd515bb6e87aad4f25dbad0a600d0c",
    urls = [
        "http://storage.googleapis.com/download.tensorflow.org/models/speech_commands_v0.01.zip",
        "http://download.tensorflow.org/models/speech_commands_v0.01.zip",
    ],
)
