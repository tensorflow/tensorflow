licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

load("//third_party/toolchains/preconfig/generate:containers.bzl", "container_digests")

# TODO(b/122347293): This is the RBE config based on the CPU configuration / image provided
# in the asci-toolchain setup. Delete this once we switched CPU remote builds to the
# new platform below.
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

# Remote build platforms.
# Each of the platform rules here provide a platform definition that is bound to a docker image.
# The result of the skylark configuration is checked into
# //tensorflow/third_party/toolchains/preconfig.

# Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cpu.
platform(
    name = "rbe_ubuntu16.04",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:linux",
    ],
    remote_execution_properties = """
        properties: {
            name: "container-image"
            value:"docker://gcr.io/tensorflow-testing/nosla-ubuntu16.04@%s"
        }""" % container_digests["ubuntu16.04"],
)

# Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda9.0-cudnn7-ubuntu14.04.
platform(
    name = "rbe_cuda9.0-cudnn7-ubuntu14.04",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:linux",
    ],
    remote_execution_properties = """
        properties: {
            name: "container-image"
            value:"docker://gcr.io/tensorflow-testing/nosla-cuda9.0-cudnn7-ubuntu14.04@%s"
        }""" % container_digests["cuda9.0-cudnn7-ubuntu14.04"],
)

# Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda10.0-cudnn7-ubuntu14.04.
platform(
    name = "rbe_cuda10.0-cudnn7-ubuntu14.04",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:linux",
    ],
    remote_execution_properties = """
        properties: {
            name: "container-image"
            value:"docker://gcr.io/tensorflow-testing/nosla-cuda10.0-cudnn7-ubuntu14.04@%s"
        }""" % container_digests["cuda10.0-cudnn7-ubuntu14.04"],
)
