load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)
load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_pull",
)
load(":containers.bzl", "container_digests")

def _remote_config_workspace():
    container_repositories()

    container_pull(
        name = "centos6",
        registry = "gcr.io",
        repository = "tensorflow-testing/nosla-centos6",
        digest = container_digests["centos6"],
    )

    container_pull(
        name = "ubuntu16.04",
        registry = "gcr.io",
        repository = "tensorflow-testing/nosla-ubuntu16.04",
        digest = container_digests["ubuntu16.04"],
    )

    container_pull(
        name = "cuda10.0-cudnn7-ubuntu14.04",
        registry = "gcr.io",
        repository = "tensorflow-testing/nosla-cuda10.0-cudnn7-ubuntu14.04",
        digest = container_digests["cuda10.0-cudnn7-ubuntu14.04"],
    )

    container_pull(
        name = "cuda10.0-cudnn7-centos6",
        registry = "gcr.io",
        repository = "tensorflow-testing/nosla-cuda10.0-cudnn7-centos6",
        digest = container_digests["cuda10.0-cudnn7-centos6"],
    )

    container_pull(
        name = "cuda10.1-cudnn7-centos6",
        registry = "gcr.io",
        repository = "tensorflow-testing/nosla-cuda10.1-cudnn7-centos6",
        digest = container_digests["cuda10.1-cudnn7-centos6"],
    )

    container_pull(
        name = "ubuntu16.04-manylinux2010",
        registry = "gcr.io",
        repository = "tensorflow-testing/nosla-ubuntu16.04-manylinux2010",
        digest = container_digests["ubuntu16.04-manylinux2010"],
    )

    container_pull(
        name = "cuda10.0-cudnn7-ubuntu16.04-manylinux2010",
        registry = "gcr.io",
        repository = "tensorflow-testing/nosla-cuda10.0-cudnn7-ubuntu16.04-manylinux2010",
        digest = container_digests["cuda10.0-cudnn7-ubuntu16.04-manylinux2010"],
    )

    container_pull(
        name = "cuda10.1-cudnn7-ubuntu16.04-manylinux2010",
        registry = "gcr.io",
        repository = "tensorflow-testing/nosla-cuda10.1-cudnn7-ubuntu16.04-manylinux2010",
        digest = container_digests["cuda10.1-cudnn7-ubuntu16.04-manylinux2010"],
    )

    container_pull(
        name = "rocm-ubuntu16.04",
        registry = "gcr.io",
        repository = "tensorflow-testing/nosla-rocm-ubuntu16.04",
        digest = container_digests["rocm-ubuntu16.04"],
    )

remote_config_workspace = _remote_config_workspace
