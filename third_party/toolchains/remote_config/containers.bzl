"""Docker images used with remote config and RBE."""

load("//third_party/toolchains/preconfig/generate:containers.bzl", "container_digests")

containers = {

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.cuda10.0-cudnn7-ubuntu16.04-manylinux2010.
    "cuda10.0-cudnn7-ubuntu16.04-manylinux2010": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-cuda10.0-cudnn7-ubuntu16.04-manylinux2010",
        "digest": container_digests["cuda10.0-cudnn7-ubuntu16.04-manylinux2010"],
    },

    # Built with //tensorflow/tools/ci_build/Dockerfile.rbe.rocm-ubuntu16.04
    "rocm-ubuntu16.04": {
        "registry": "gcr.io",
        "repository": "tensorflow-testing/nosla-rocm-ubuntu16.04",
        "digest": container_digests["rocm-ubuntu16.04"],
    },
}
