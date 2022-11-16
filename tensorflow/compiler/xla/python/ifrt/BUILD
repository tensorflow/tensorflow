load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/xla/python/...",
    ],
)

package(
    default_visibility = [":friends"],
    licenses = ["notice"],
)

cc_library(
    name = "ifrt",
    srcs = [
        "array.cc",
        "client.cc",
        "compiler.cc",
        "dtype.cc",
        "executable.cc",
        "future.cc",
        "shape.cc",
        "sharding.cc",
    ],
    hdrs = [
        "array.h",
        "client.h",
        "compiler.h",
        "device.h",
        "dtype.h",
        "executable.h",
        "future.h",
        "shape.h",
        "sharding.h",
    ],
    deps = [
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
    ],
)

tf_cc_test(
    name = "array_test",
    size = "small",
    srcs = ["array_test.cc"],
    deps = [
        ":ifrt",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)

tf_cc_test(
    name = "future_test",
    size = "small",
    srcs = ["future_test.cc"],
    deps = [
        ":ifrt",
        "//tensorflow/tsl/lib/core:status_test_util",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "shape_test",
    size = "small",
    srcs = ["shape_test.cc"],
    deps = [
        ":ifrt",
        "@com_google_googletest//:gtest_main",
    ],
)

tf_cc_test(
    name = "sharding_test",
    size = "small",
    srcs = ["sharding_test.cc"],
    deps = [
        ":ifrt",
        "//tensorflow/tsl/lib/core:status_test_util",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//llvm:Support",
    ],
)
