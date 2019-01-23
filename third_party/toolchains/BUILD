licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

load("//third_party/toolchains/preconfig/generate:containers.bzl", "container_digests")

# Constraint used for platforms below so we can force certain rules to be executed
# on specific platforms.
constraint_setting(name = "custom_platforms")

# Constraint for platforms that allow GPU testing (i.e. have a GPU available).
# This is used in exec_compatible_with of rules that need GPU access.
constraint_value(
    name = "gpu_test",
    constraint_setting = ":custom_platforms",
)

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

# The above platform with GPU support.
platform(
    name = "rbe_cuda10.0-cudnn7-ubuntu14.04-gpu",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:linux",
        ":gpu_test",
    ],
    remote_execution_properties = """
        properties: {
            name: "container-image"
            value: "docker://gcr.io/tensorflow-testing/nosla-cuda10.0-cudnn7-ubuntu14.04@%s"
        }
        properties: {
            name: "dockerRuntime"
            value: "nvidia"
        }
        properties: {
            name: "Pool"
            value: "gpu-pool"
        }
        """ % container_digests["cuda10.0-cudnn7-ubuntu14.04"],
)
