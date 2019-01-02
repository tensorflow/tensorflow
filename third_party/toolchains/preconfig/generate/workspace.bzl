load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_pull",
    container_repositories = "repositories",
)
load(":containers.bzl", "container_digests")

def _remote_config_workspace():
    container_repositories()

    container_pull(
        name = "cuda9.0-cudnn7-ubuntu14.04",
        registry = "gcr.io",
        repository = "asci-toolchain/nosla-cuda9.0-cudnn7-ubuntu14.04",
        digest = container_digests["cuda9.0-cudnn7-ubuntu14.04"],
    )

    container_pull(
        name = "cuda10.0-cudnn7-ubuntu14.04",
        registry = "gcr.io",
        repository = "tensorflow-testing/nosla-cuda10.0-cudnn7-ubuntu14.04",
        digest = container_digests["cuda10.0-cudnn7-ubuntu14.04"],
    )

remote_config_workspace = _remote_config_workspace
