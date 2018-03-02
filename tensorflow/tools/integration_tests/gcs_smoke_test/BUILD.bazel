package(default_visibility = ["//visibility:public"])

load("@rbe_integration_test//skylark:integration_tests.bzl", "sut_component", "integration_test")
load("@rbe_integration_test//skylark:toolchains.bzl", "toolchain_container_images")

sut_component(
    name = "gcs",
    docker_image = toolchain_container_images()["tensorflow"],
    setups = [{
        "program": "setup.sh",
        "args": [
            "gs://tensorflow-test-bucket/tf-gcs-test",
        ],
        "output_properties": ["gcs_path"],
        "timeout_seconds": 100,
    }],
    teardowns = [{
        "program": "teardown.sh",
        "args": ["{gcs_path}"],
        "timeout_seconds": 100,
    }],
)

py_binary(
    name = "gcs_smoke",
    srcs = ["gcs_smoke.py"],
)

sh_binary(
    name = "test_wrapper",
    srcs = ["test_wrapper.sh"],
    data = [
        "gcs_smoke",
    ],
)

integration_test(
    name = "gcs_smoke_test",
    sut_deps = {
        ":gcs": "gcs",
    },
    tags = [
        "manual",
        "notap",
    ],
    test = {
        "program": ":test_wrapper",
        "args": [
            "--gcs_bucket_url={gcs#gcs_path}",
            "--num_examples=20",
        ],
        "timeout_seconds": 250,
    },
    test_docker_image = toolchain_container_images()["tensorflow"],
    test_type = "MultiMachine",
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)
