load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

BASE_COPTS = [
    "-fexceptions",
    "-Wno-unused-variable",
    "-Wno-ctad-maybe-unsupported",
    "-Wno-reorder-ctor",
    "-Wno-non-virtual-dtor",
    "-Wno-uninitialized",
    "-Wno-pass-failed",
    "-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE",
    "-DRAFT_SYSTEM_LITTLE_ENDIAN",
]

cuda_library(
    name = "raft_matrix",
    srcs = glob([
        "cpp/include/raft/core/detail/*.hpp",
        "cpp/include/raft/core/resource/detail/*.hpp",
        "cpp/include/raft/linalg/detail/*.hpp",
        "cpp/include/raft/matrix/detail/*.hpp",
        "cpp/include/raft/util/detail/*.hpp",
    ]),
    hdrs = glob([
        "cpp/include/raft/core/*.hpp",
        "cpp/include/raft/core/resource/*.hpp",
        "cpp/include/raft/linalg/*.hpp",
        "cpp/include/raft/matrix/*.hpp",
        "cpp/include/raft/util/*.hpp",
    ]),
    copts = BASE_COPTS,
    includes = ["cpp/include"],
    textual_hdrs = glob([
        "cpp/include/raft/core/**/*.cuh",
        "cpp/include/raft/linalg/**/*.cuh",
        "cpp/include/raft/matrix/**/*.cuh",
        "cpp/include/raft/util/**/*.cuh",
    ]),
    visibility = ["//visibility:public"],
    deps = [
        "@kokkos//:mdspan",
        "@rapids_logger",
        "@rmm",
    ],
)

cuda_library(
    name = "select_k_runner",
    srcs = ["select_k_runner.cu.cc"],
    hdrs = ["select_k_runner.hpp"],
    copts = BASE_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":raft_matrix",
    ],
)

cc_test(
    name = "select_k_smoke_test",
    srcs = ["select_k_smoke_test.cu.cc"],
    copts = BASE_COPTS,
    deps = [
        ":select_k_runner",
        "@com_google_googletest//:gtest_main",
    ],
)
