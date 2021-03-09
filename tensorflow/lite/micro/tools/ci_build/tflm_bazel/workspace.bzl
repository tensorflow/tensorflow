# TensorFlow external dependencies that can be loaded in WORKSPACE files.

load("//tensorflow:version_check.bzl", "check_bazel_version_at_least")
load("//third_party/clog:workspace.bzl", clog = "repo")
load("//third_party/cpuinfo:workspace.bzl", cpuinfo = "repo")
load("//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
load("//third_party/kissfft:workspace.bzl", kissfft = "repo")
load("//third_party/ruy:workspace.bzl", ruy = "repo")
load("//third_party:repo.bzl", "tf_http_archive")

def initialize_third_party():
    """ Load third party repositories.  See above load() statements. """
    clog()
    cpuinfo()
    flatbuffers()
    kissfft()
    ruy()

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def tf_repositories(path_prefix = "", tf_repo_name = ""):
    """All external dependencies for TF builds."""

    tf_http_archive(
        name = "arm_neon_2_x86_sse",
        build_file = clean_dep("//third_party:arm_neon_2_x86_sse.BUILD"),
        sha256 = "213733991310b904b11b053ac224fee2d4e0179e46b52fe7f8735b8831e04dcc",
        strip_prefix = "ARM_NEON_2_x86_SSE-1200fe90bb174a6224a525ee60148671a786a71f",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/intel/ARM_NEON_2_x86_SSE/archive/1200fe90bb174a6224a525ee60148671a786a71f.tar.gz",
            "https://github.com/intel/ARM_NEON_2_x86_SSE/archive/1200fe90bb174a6224a525ee60148671a786a71f.tar.gz",
        ],
    )

    # https://github.com/bazelbuild/bazel-skylib/releases
    tf_http_archive(
        name = "bazel_skylib",
        sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
        ],
    )

    tf_http_archive(
        name = "build_bazel_rules_android",
        sha256 = "cd06d15dd8bb59926e4d65f9003bfc20f9da4b2519985c27e190cddc8b7a7806",
        strip_prefix = "rules_android-0.1.1",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_android/archive/v0.1.1.zip",
            "https://github.com/bazelbuild/rules_android/archive/v0.1.1.zip",
        ],
    )

    EIGEN_COMMIT = "90ee821c563fa20db4d64d6991ddca256d5c52f2"
    EIGEN_SHA256 = "d76992f1972e4ff270221c7ee8125610a8e02bb46708a7295ee646e99287083b"

    tf_http_archive(
        name = "eigen_archive",
        build_file = "//third_party/eigen3:eigen_archive.BUILD",
        sha256 = EIGEN_SHA256,
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
            "https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
        ],
    )

    tf_http_archive(
        name = "gemmlowp",
        sha256 = "43146e6f56cb5218a8caaab6b5d1601a083f1f31c06ff474a4378a7d35be9cfb",  # SHARED_GEMMLOWP_SHA
        strip_prefix = "gemmlowp-fda83bdc38b118cc6b56753bd540caa49e570745",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/gemmlowp/archive/fda83bdc38b118cc6b56753bd540caa49e570745.zip",
            "https://github.com/google/gemmlowp/archive/fda83bdc38b118cc6b56753bd540caa49e570745.zip",
        ],
    )

    initialize_third_party()

def workspace():
    # Check the bazel version before executing any repository rules, in case
    # those rules rely on the version we require here.
    check_bazel_version_at_least("1.0.0")
    tf_repositories()
