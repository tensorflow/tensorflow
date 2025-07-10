load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/tools/evaluation/tasks:build_def.bzl", "task_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "fully_connected_drivers",
    srcs = [
        "fully_connected_bram_driver.cc",
        "fully_connected_ip_driver.cc",
    ],
    hdrs = [
        "fully_connected_bram_driver.h",
        "fully_connected_ip_driver.h",
    ],
    copts = tflite_copts(),
    alwayslink = 1,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "fully_connected_fpga_test_bram",
    srcs = ["main.cpp"],
    deps = [":fully_connected_drivers"],
    linkopts = ["-lrt", "-lpthread"],
    copts = tflite_copts(),
)

cc_binary(
    name = "fully_connected_fpga_test_ip",
    srcs = ["main_ip_test.cpp"],
    deps = [":fully_connected_drivers"],
    linkopts = ["-lrt", "-lpthread"],
    copts = tflite_copts(),
)
