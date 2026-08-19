load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

BASE_COPTS = [
    "-Wno-unused-result",
    "-Wno-ctad-maybe-unsupported",
    "-Wno-self-move",
    "-Wno-pragma-once-outside-header",
    "-fexceptions",
    "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE",
]

cuda_library(
    name = "rmm",
    srcs = glob([
        "cpp/src/*.cpp",
    ]),
    copts = BASE_COPTS,
    features = ["-use_header_modules"],
    includes = ["cpp/include"],
    textual_hdrs = glob([
        "cpp/include/rmm/**/*.h",
        "cpp/include/rmm/**/*.hpp",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@rapids_logger",
        "@xla//xla/tsl/cuda:cudart",
    ],
)

cc_test(
    name = "cuda_stream_tests",
    srcs = ["cpp/tests/cuda_stream_tests.cpp"],
    copts = BASE_COPTS,
    deps = [
        ":rmm",
        "@com_google_googletest//:gtest_main",
    ],
)
