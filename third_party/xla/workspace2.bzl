"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

# Import third party config rules.
load("@bazel_skylib//lib:versions.bzl", "versions")

# Import TSL Workspaces
load("@local_tsl//:workspace2.bzl", "tsl_workspace2")
load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("//third_party/dlpack:workspace.bzl", dlpack = "repo")

# Import third party repository rules. See go/tfbr-thirdparty.
load("//third_party/FP16:workspace.bzl", FP16 = "repo")
load("//third_party/gloo:workspace.bzl", gloo = "repo")
load("//third_party/mpitrampoline:workspace.bzl", mpitrampoline = "repo")
load("//third_party/nanobind:workspace.bzl", nanobind = "repo")
load("//third_party/robin_map:workspace.bzl", robin_map = "repo")
load("//third_party/shardy:workspace.bzl", shardy = "repo")
load("//third_party/stablehlo:workspace.bzl", stablehlo = "repo")
load("//third_party/triton:workspace.bzl", triton = "repo")
load("//third_party/uv:workspace.bzl", uv = "repo")

def _initialize_third_party():
    """ Load third party repositories.  See above load() statements. """
    FP16()
    dlpack()
    gloo()
    mpitrampoline()
    nanobind()
    robin_map()
    shardy()
    stablehlo()
    triton()
    uv()

# Define all external repositories required by TensorFlow
def _tf_repositories():
    """All external dependencies for TF builds."""

    # To update any of the dependencies below:
    # a) update URL and strip_prefix to the new git commit hash
    # b) get the sha256 hash of the commit by running:
    #    curl -L <url> | sha256sum
    # and update the sha256 with the result.

    # LINT.IfChange
    tf_http_archive(
        name = "XNNPACK",
        sha256 = "3306f4178c8594b689165d385e644f03a3154c3be044f6ae36dd170fbf182cf5",
        strip_prefix = "XNNPACK-983d013300f19fd3f4e33220b6401408e97a8d12",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/983d013300f19fd3f4e33220b6401408e97a8d12.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)

    tf_http_archive(
        name = "KleidiAI",
        sha256 = "ad37707084a6d4ff41be10cbe8540c75bea057ba79d0de6c367c1bfac6ba0852",
        strip_prefix = "kleidiai-40a926833857fb64786e02f97703e42b1537cb57",
        urls = tf_mirror_urls("https://gitlab.arm.com/kleidi/kleidiai/-/archive/40a926833857fb64786e02f97703e42b1537cb57/kleidiai-40a926833857fb64786e02f97703e42b1537cb57.zip"),
    )

    tf_http_archive(
        name = "FXdiv",
        sha256 = "3d7b0e9c4c658a84376a1086126be02f9b7f753caa95e009d9ac38d11da444db",
        strip_prefix = "FXdiv-63058eff77e11aa15bf531df5dd34395ec3017c8",
        urls = tf_mirror_urls("https://github.com/Maratyszcza/FXdiv/archive/63058eff77e11aa15bf531df5dd34395ec3017c8.zip"),
    )

    tf_http_archive(
        name = "cpuinfo",
        sha256 = "52e0ffd7998d8cb3a927d8a6e1145763744d866d2be09c4eccea27fc157b6bb0",
        strip_prefix = "cpuinfo-cebb0933058d7f181c979afd50601dc311e1bf8c",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/cebb0933058d7f181c979afd50601dc311e1bf8c.zip"),
    )

    tf_http_archive(
        name = "pthreadpool",
        sha256 = "a4cf06de57bfdf8d7b537c61f1c3071bce74e57524fe053e0bbd2332feca7f95",
        strip_prefix = "pthreadpool-4fe0e1e183925bf8cfa6aae24237e724a96479b8",
        urls = tf_mirror_urls("https://github.com/Maratyszcza/pthreadpool/archive/4fe0e1e183925bf8cfa6aae24237e724a96479b8.zip"),
    )

    tf_http_archive(
        name = "jsoncpp_git",
        sha256 = "f409856e5920c18d0c2fb85276e24ee607d2a09b5e7d5f0a371368903c275da2",
        strip_prefix = "jsoncpp-1.9.5",
        system_build_file = "//third_party/systemlibs:jsoncpp.BUILD",
        urls = tf_mirror_urls("https://github.com/open-source-parsers/jsoncpp/archive/1.9.5.tar.gz"),
    )

    tf_http_archive(
        name = "cudnn_frontend_archive",
        build_file = "//third_party:cudnn_frontend.BUILD",
        patch_file = ["//third_party:cudnn_frontend_header_fix.patch"],
        sha256 = "5f77784dc3ccbca7aca5ea0b5a6e31b95aa85023c5942d22be5fa8dd6c339d81",
        strip_prefix = "cudnn-frontend-1.8.0",
        urls = tf_mirror_urls("https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v1.8.0.zip"),
    )

    tf_http_archive(
        name = "cutlass_archive",
        build_file = "//third_party:cutlass.BUILD",
        sha256 = "84cf3fcc47c440a8dde016eb458f8d6b93b3335d9c3a7a16f388333823f1eae0",
        strip_prefix = "cutlass-afa7b7241aabe598b725c65480bd9fa71121732c",
        urls = tf_mirror_urls("https://github.com/chsigg/cutlass/archive/afa7b7241aabe598b725c65480bd9fa71121732c.tar.gz"),
    )

    tf_http_archive(
        name = "boringssl",
        sha256 = "9dc53f851107eaf87b391136d13b815df97ec8f76dadb487b58b2fc45e624d2c",
        strip_prefix = "boringssl-c00d7ca810e93780bd0c8ee4eea28f4f2ea4bcdc",
        system_build_file = "//third_party/systemlibs:boringssl.BUILD",
        urls = tf_mirror_urls("https://github.com/google/boringssl/archive/c00d7ca810e93780bd0c8ee4eea28f4f2ea4bcdc.tar.gz"),
    )

    tf_http_archive(
        name = "com_google_ortools",
        sha256 = "bc4b07dc9c23f0cca43b1f5c889f08a59c8f2515836b03d4cc7e0f8f2c879234",
        strip_prefix = "or-tools-9.6",
        patch_file = ["//third_party/ortools:ortools.patch"],
        urls = tf_mirror_urls("https://github.com/google/or-tools/archive/v9.6.tar.gz"),
        repo_mapping = {
            "@com_google_protobuf_cc": "@com_google_protobuf",
            "@eigen": "@eigen_archive",
        },
    )

    tf_http_archive(
        name = "glpk",
        sha256 = "9a5dab356268b4f177c33e00ddf8164496dc2434e83bd1114147024df983a3bb",
        build_file = "//third_party/ortools:glpk.BUILD",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/ftp.gnu.org/gnu/glpk/glpk-4.52.tar.gz",
            "http://ftp.gnu.org/gnu/glpk/glpk-4.52.tar.gz",
        ],
    )

    tf_http_archive(
        name = "scip",
        sha256 = "fe7636f8165a8c9298ff55ed3220d084d4ea31ba9b69d2733beec53e0e4335d6",
        strip_prefix = "scip-803",
        build_file = "//third_party/ortools:scip.BUILD",
        patch_file = ["//third_party/ortools:scip.patch"],
        urls = tf_mirror_urls("https://github.com/scipopt/scip/archive/refs/tags/v803.tar.gz"),
    )

    tf_http_archive(
        name = "bliss",
        build_file = "//third_party/ortools:bliss.BUILD",
        sha256 = "f57bf32804140cad58b1240b804e0dbd68f7e6bf67eba8e0c0fa3a62fd7f0f84",
        urls = tf_mirror_urls("https://github.com/google/or-tools/releases/download/v9.0/bliss-0.73.zip"),
        #url = "http://www.tcs.hut.fi/Software/bliss/bliss-0.73.zip",
    )

    tf_http_archive(
        name = "pybind11_protobuf",
        urls = tf_mirror_urls("https://github.com/pybind/pybind11_protobuf/archive/80f3440cd8fee124e077e2e47a8a17b78b451363.zip"),
        sha256 = "c7ab64b1ccf9a678694a89035a8c865a693e4e872803778f91f0965c2f281d78",
        strip_prefix = "pybind11_protobuf-80f3440cd8fee124e077e2e47a8a17b78b451363",
    )

# buildifier: disable=function-docstring
# buildifier: disable=unnamed-macro
def workspace():
    tsl_workspace2()

    # Check the bazel version before executing any repository rules, in case
    # those rules rely on the version we require here.
    versions.check("1.0.0")

    # Import third party repositories according to go/tfbr-thirdparty.
    _initialize_third_party()

    # Import all other repositories. This should happen before initializing
    # any external repositories, because those come with their own
    # dependencies. Those recursive dependencies will only be imported if they
    # don't already exist (at least if the external repository macros were
    # written according to common practice to query native.existing_rule()).
    _tf_repositories()

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
xla_workspace2 = workspace
