load("//tensorflow/tools/toolchains/remote_config:containers.bzl", "container_digests")

licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

# Remote build platforms.
# Each of the platform rules here provide a platform definition that is bound to a docker image.
# The result of the starlark configuration is checked into
# //third_party/toolchains/preconfig.

# Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cpu.
platform(
    name = "rbe_ubuntu16.04",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:linux",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/nosla-ubuntu16.04@%s" % container_digests["ubuntu16.04"],
        "Pool": "default",
    },
)

# Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cpu-centos6.
platform(
    name = "rbe_centos6",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:linux",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/nosla-centos6@%s" % container_digests["centos6"],
        "Pool": "default",
    },
)

# Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda10.0-cudnn7-ubuntu14.04.
platform(
    name = "rbe_cuda10.0-cudnn7-ubuntu14.04",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:linux",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/nosla-cuda10.0-cudnn7-ubuntu14.04@%s" % container_digests["cuda10.0-cudnn7-ubuntu14.04"],
        "Pool": "default",
    },
)

# Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda10.0-cudnn7-centos6.
platform(
    name = "rbe_cuda10.0-cudnn7-centos6",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:linux",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/nosla-cuda10.0-cudnn7-centos6@%s" % container_digests["cuda10.0-cudnn7-centos6"],
        "Pool": "default",
    },
)

# Built with //tensorflow/tools/ci_build/Dockerfile.rbe.ubuntu16.04-manylinux2010.
platform(
    name = "rbe_ubuntu16.04-manylinux2010",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:linux",
    ],
    exec_properties = {
        "container-image": "docker://gcr.io/tensorflow-testing/nosla-ubuntu16.04-manylinux2010@%s" % container_digests["ubuntu16.04-manylinux2010"],
        "Pool": "default",
    },
)
