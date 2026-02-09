workspace(name = "org_tensorflow_text")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//third_party/bazel:http.bzl", "custom_http_archive")

# githu.com/tensorflow#104948 - Support latest Xcode.
http_archive(
    name = "build_bazel_apple_support",
    sha256 = "1ae6fcf983cff3edab717636f91ad0efff2e5ba75607fdddddfd6ad0dbdfaf10",
    urls = ["https://github.com/bazelbuild/apple_support/releases/download/1.24.5/apple_support.1.24.5.tar.gz"],
)

custom_http_archive(
    name = "icu",
    exclude = [
        # Build file does not contain all required targets.
        "icu4c/source/common/BUILD.bazel",
    ],
    files = {
        "BUILD.bazel": "//third_party/icu:BUILD.bzl",
    },
    sha256 = "e424ba5282d95ad38b52639a08fb82164f0b0cbd7f17b53ae16bf14f8541855f",
    strip_prefix = "icu-release-77-1",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/unicode-org/icu/archive/release-77-1.zip",
        "https://github.com/unicode-org/icu/archive/release-77-1.zip",
    ],
)

http_archive(
    name = "com_google_sentencepiece",
    build_file = "//third_party/sentencepiece:BUILD",
    patch_args = ["-p1"],
    patches = ["//third_party/sentencepiece:sp.patch"],
    sha256 = "8409b0126ebd62b256c685d5757150cf7fcb2b92a2f2b98efb3f38fc36719754",
    strip_prefix = "sentencepiece-0.1.96",
    urls = [
        "https://github.com/google/sentencepiece/archive/refs/tags/v0.1.96.zip",
    ],
)

http_archive(
    name = "cppitertools",
    sha256 = "e56741b108d6baced98c4ccd83fd0d5a545937f2845978799c28d0312c0dee3d",
    strip_prefix = "cppitertools-2.0",
    urls = ["https://github.com/ryanhaining/cppitertools/archive/refs/tags/v2.0.zip"],
)

http_archive(
    name = "darts_clone",
    build_file = "//third_party/darts_clone:BUILD.bzl",
    sha256 = "c97f55d05c98da6fcaf7f9ecc6a6dc6bc5b18b8564465f77abff8879d446491c",
    strip_prefix = "darts-clone-e40ce4627526985a7767444b6ed6893ab6ff8983",
    urls = [
        "https://github.com/s-yata/darts-clone/archive/e40ce4627526985a7767444b6ed6893ab6ff8983.zip",
    ],
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
    strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)

http_archive(
    name = "org_tensorflow",
    sha256 = "213edf03ac7c4e74d8eb2074216ae8c8ae4f325c6bc22efd16cfdeed2073bd66",
    strip_prefix = "tensorflow-2.20.0",
    patch_args = ["-p1"],
    patches = ["//third_party/tensorflow:tensorflow.core.BUILD.patch"],
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.20.0.zip",
    ],
)

http_archive(
    name = "org_tensorflow_datasets",
    sha256 = "c6ff4e2306387f0ca45d4f616d9a1c5e79e02ef16d0a8958230a8049ea07fc98",
    strip_prefix = "datasets-3.2.0",
    urls = [
        "https://github.com/tensorflow/datasets/archive/v3.2.0.zip",
    ],
)

http_archive(
    name = "pybind11",
    build_file = "//third_party/pybind11:BUILD.bzl",
    sha256 = "efc901aa0aab439a3fea6efeaf930b5a349fb06394bf845c64ce15a9cf8f0240",
    strip_prefix = "pybind11-2.13.4",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/pybind/pybind11/archive/v2.13.4.tar.gz",
        "https://github.com/pybind/pybind11/archive/v2.13.4.tar.gz",
    ],
)

http_archive(
    name = "rules_shell",
    sha256 = "bc61ef94facc78e20a645726f64756e5e285a045037c7a61f65af2941f4c25e1",
    strip_prefix = "rules_shell-0.4.1",
    url = "https://github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
)

# Initialize hermetic Python
load("@org_tensorflow//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("//third_party/bazel:py_deps_profile.bzl", "py_deps_profile")

py_deps_profile(
    name = "release_or_nightly",
    deps_map = {
        "tensorflow": [
            "tf-nightly",
            "tf_headers",
            "tf_header_lib",
            "libtensorflow_framework",
        ],
        "tf-keras": ["tf-keras-nightly"],
    },
    pip_repo_name = "pypi",
    requirements_in = "//oss_scripts/pip_package:requirements.in",
    switch = {
        "IS_NIGHTLY": "nightly",
    },
)

load("@org_tensorflow//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    requirements = {
        "3.9": "//oss_scripts/pip_package:requirements_lock_3_9.txt",
        "3.10": "//oss_scripts/pip_package:requirements_lock_3_10.txt",
        "3.11": "//oss_scripts/pip_package:requirements_lock_3_11.txt",
        "3.12": "//oss_scripts/pip_package:requirements_lock_3_12.txt",
        "3.13": "//oss_scripts/pip_package:requirements_lock_3_13.txt",
    },
)

load("@org_tensorflow//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("//third_party/tensorflow:tf_configure.bzl", "tf_configure")

tf_configure()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

# Initialize TensorFlow dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

# Set up Android.
load("@org_tensorflow//third_party/android:android_configure.bzl", "android_configure")

android_configure(name = "local_config_android")

load("@local_config_android//:android.bzl", "android_workspace")

android_workspace()

load(
    "@local_xla//third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)

python_wheel_version_suffix_repository(name = "tf_wheel_version_suffix")

load(
    "@local_xla//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@local_xla//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@local_xla//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@local_xla//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@local_xla//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")
