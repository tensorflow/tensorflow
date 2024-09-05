# Description:
#  Holds model-agnostic files and proto definitions. The app will bundle the files into assets.

load("//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android:proto.bzl", "proto_data")
# copybara:uncomment load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/lite/tools/benchmark/experimental/delegate_performance/android:__subpackages__"],
    licenses = ["notice"],
)

proto_library(
    name = "delegate_performance_proto",
    srcs = ["delegate_performance.proto"],
)

cc_proto_library(
    name = "delegate_performance_cc_proto",
    deps = [":delegate_performance_proto"],
)

java_lite_proto_library(
    name = "delegate_performance_java_proto_lite",
    deps = [":delegate_performance_proto"],
)

proto_data(
    name = "default_latency_criteria",
    src = "default_latency_criteria.textproto",
    out = "assets/proto/default_latency_criteria.binarypb",
    proto_deps = [":delegate_performance_proto"],
    proto_name = "tflite.proto.benchmark.LatencyCriteria",
)
