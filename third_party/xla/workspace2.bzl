"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""

# Import third party config rules.
load("@bazel_skylib//lib:versions.bzl", "versions")

# Import TSL Workspaces
load("//:tsl_workspace2.bzl", "tsl_workspace2")
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
        sha256 = "763f32eed3520cdb2fc906555315701ac4cc92e3d77acd4a3eafd8a553996d63",
        strip_prefix = "XNNPACK-0a655ef53812ce9bd8ce2628757cc0f476efcf51",
        urls = tf_mirror_urls("https://github.com/google/XNNPACK/archive/0a655ef53812ce9bd8ce2628757cc0f476efcf51.zip"),
    )
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/xnnpack.cmake)

    tf_http_archive(
        name = "KleidiAI",
        sha256 = "439926527fca9405ae90b602a3938d3435751ec78492e5f1c62d85f5df8c2784",
        strip_prefix = "kleidiai-dc69e899945c412a8ce39ccafd25139f743c60b1",
        urls = tf_mirror_urls("https://github.com/ARM-software/kleidiai/archive/dc69e899945c412a8ce39ccafd25139f743c60b1.zip"),
    )

    tf_http_archive(
        name = "FXdiv",
        sha256 = "3d7b0e9c4c658a84376a1086126be02f9b7f753caa95e009d9ac38d11da444db",
        strip_prefix = "FXdiv-63058eff77e11aa15bf531df5dd34395ec3017c8",
        urls = tf_mirror_urls("https://github.com/Maratyszcza/FXdiv/archive/63058eff77e11aa15bf531df5dd34395ec3017c8.zip"),
    )

    tf_http_archive(
        name = "cpuinfo",
        sha256 = "ae356c4c0c841e20711b5e111a1ccdec9c2f3c1dd7bde7cfba1bed18d6d02459",
        strip_prefix = "cpuinfo-de0ce7c7251372892e53ce9bc891750d2c9a4fd8",
        urls = tf_mirror_urls("https://github.com/pytorch/cpuinfo/archive/de0ce7c7251372892e53ce9bc891750d2c9a4fd8.zip"),
    )

    tf_http_archive(
        name = "pthreadpool",
        sha256 = "516ba8d05c30e016d7fd7af6a7fc74308273883f857faf92bc9bb630ab6dba2c",
        strip_prefix = "pthreadpool-c2ba5c50bb58d1397b693740cf75fad836a0d1bf",
        urls = tf_mirror_urls("https://github.com/google/pthreadpool/archive/c2ba5c50bb58d1397b693740cf75fad836a0d1bf.zip"),
    )

    tf_http_archive(
        name = "jsoncpp_git",
        sha256 = "f409856e5920c18d0c2fb85276e24ee607d2a09b5e7d5f0a371368903c275da2",
        strip_prefix = "jsoncpp-1.9.5",
        urls = tf_mirror_urls("https://github.com/open-source-parsers/jsoncpp/archive/1.9.5.tar.gz"),
    )

    tf_http_archive(
        name = "cudnn_frontend_archive",
        build_file = "//third_party:cudnn_frontend.BUILD",
        patch_file = ["//third_party:cudnn_frontend_header_fix.patch"],
        sha256 = "34dfe01057e43e799af207522aa0c863ad3177f8c1568b6e7a7e4ccf1cbff769",
        strip_prefix = "cudnn-frontend-1.11.0",
        urls = tf_mirror_urls("https://github.com/NVIDIA/cudnn-frontend/archive/refs/tags/v1.11.0.zip"),
    )

    tf_http_archive(
        name = "cutlass_archive",
        build_file = "//third_party:cutlass.BUILD",
        sha256 = "a7739ca3dc74e3a5cb57f93fc95224c5e2a3c2dff2c16bb09a5e459463604c08",
        strip_prefix = "cutlass-3.8.0",
        urls = tf_mirror_urls("https://github.com/NVIDIA/cutlass/archive/refs/tags/v3.8.0.zip"),
    )

    tf_http_archive(
        name = "boringssl",
        sha256 = "9dc53f851107eaf87b391136d13b815df97ec8f76dadb487b58b2fc45e624d2c",
        strip_prefix = "boringssl-c00d7ca810e93780bd0c8ee4eea28f4f2ea4bcdc",
        urls = tf_mirror_urls("https://github.com/google/boringssl/archive/c00d7ca810e93780bd0c8ee4eea28f4f2ea4bcdc.tar.gz"),
    )

    tf_http_archive(
        name = "com_google_ortools",
        sha256 = "f6a0bd5b9f3058aa1a814b798db5d393c31ec9cbb6103486728997b49ab127bc",
        strip_prefix = "or-tools-9.11",
        patch_file = ["//third_party/ortools:ortools.patch"],
        urls = tf_mirror_urls("https://github.com/google/or-tools/archive/v9.11.tar.gz"),
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
        sha256 = "ee221bd13a6b24738f2e74321e2efdebd6d7c603574ca6f6cb9d4472ead2c22f",
        strip_prefix = "scip-900",
        build_file = "@com_google_ortools//bazel:scip.BUILD.bazel",
        patch_file = ["@com_google_ortools//bazel:scip-v900.patch"],
        urls = tf_mirror_urls("https://github.com/scipopt/scip/archive/refs/tags/v900.tar.gz"),
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
        urls = tf_mirror_urls("https://github.com/pybind/pybind11_protobuf/archive/f02a2b7653bc50eb5119d125842a3870db95d251.zip"),
        sha256 = "3cf7bf0f23954c5ce6c37f0a215f506efa3035ca06e3b390d67f4cbe684dce23",
        strip_prefix = "pybind11_protobuf-f02a2b7653bc50eb5119d125842a3870db95d251",
    )

    tf_http_archive(
        name = "com_github_nelhage_rules_boost",
        urls = tf_mirror_urls("https://github.com/nelhage/rules_boost/archive/5160325dbdc8c9e499f9d9917d913f35f1785d52.zip"),
        strip_prefix = "rules_boost-5160325dbdc8c9e499f9d9917d913f35f1785d52",
        sha256 = "feb4b1294684c79df7c1e08f1aec5da0da52021e33db59c88edbe86b4d1a017a",
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
