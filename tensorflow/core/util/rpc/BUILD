load("//tensorflow:tensorflow.bzl", "tf_cc_test")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "call_container",
    hdrs = ["call_container.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
    ],
)

cc_library(
    name = "rpc_factory",
    srcs = ["rpc_factory.cc"],
    hdrs = ["rpc_factory.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "rpc_factory_registry",
    srcs = ["rpc_factory_registry.cc"],
    hdrs = ["rpc_factory_registry.h"],
    deps = [
        ":rpc_factory",
        "//tensorflow/core:framework",
    ],
)

tf_cc_test(
    name = "rpc_factory_registry_test",
    srcs = ["rpc_factory_registry_test.cc"],
    deps = [
        ":rpc_factory_registry",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
        "//tensorflow/core:testlib",
    ],
)
