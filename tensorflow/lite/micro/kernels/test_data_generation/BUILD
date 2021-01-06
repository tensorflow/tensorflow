load("@bazel_skylib//rules:build_test.bzl", "build_test")

licenses(["notice"])

cc_binary(
    name = "generate_flexbuffers_data",
    srcs = [
        "generate_flexbuffers_data.cc",
    ],
    deps = [
        "@flatbuffers",
    ],
)

build_test(
    name = "build_test",
    targets = [":generate_flexbuffers_data"],
)
