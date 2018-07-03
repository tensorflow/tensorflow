licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

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
            value:"docker://gcr.io/asci-toolchain/nosla-ubuntu16_04-tf@sha256:800a7b68cabef15419695c188ed33ed70adf678c2371b97b236f3ae26c38274d"
        }""",
)
