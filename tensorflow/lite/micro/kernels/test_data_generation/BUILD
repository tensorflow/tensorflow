load("@bazel_skylib//rules:build_test.bzl", "build_test")

package(
    features = ["-layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_binary(
    name = "generate_detection_postprocess_flexbuffers_data",
    srcs = [
        "generate_detection_postprocess_flexbuffers_data.cc",
    ],
    deps = [
        "@flatbuffers",
    ],
)

cc_binary(
    name = "generate_circular_buffer_flexbuffers_data",
    srcs = [
        "generate_circular_buffer_flexbuffers_data.cc",
    ],
    deps = [
        "@flatbuffers",
    ],
)

build_test(
    name = "build_test",
    targets = [
        ":generate_circular_buffer_flexbuffers_data",
        ":generate_detection_postprocess_flexbuffers_data",
    ],
)
