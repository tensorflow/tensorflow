licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

load("//third_party/toolchains/preconfig/generate:containers.bzl", "container_digests")

# Platform for use with remote execution with
# custom container based off RBE Ubuntu16_04
# http://gcr.io/cloud-marketplace/google/rbe-ubuntu16-04
# Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cpu
platform(
    name = "rbe_ubuntu16_04-tf",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:linux",
        "@bazel_tools//tools/cpp:clang",
        "@bazel_toolchains//constraints:xenial",
    ],
    remote_execution_properties = """
        properties: {
            name: "container-image"
            value:"docker://gcr.io/asci-toolchain/nosla-ubuntu16_04-tf@sha256:63a0e981a4e7ce5da2a851cf063e430f72947fd999d9336b7e54e2eebe8e0bf5"
        }""",
)

platform(
    name = "rbe_cuda9.0-cudnn7-ubuntu14.04",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:linux",
    ],
    remote_execution_properties = """
        properties: {
            name: "container-image"
            value:"docker://gcr.io/asci-toolchain/nosla-cuda9.0-cudnn7-ubuntu14.04@%s"
        }""" % container_digests["cuda9.0-cudnn7-ubuntu14.04"],
)
