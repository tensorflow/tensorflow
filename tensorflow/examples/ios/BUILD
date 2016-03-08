# Description:
#   Tensorflow camera demo app for Android.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//google/protobuf:protobuf.bzl", "cc_proto_library")
load("//google/protobuf:protobuf.bzl", "py_proto_library")

objc_library(
    name = "RunModelClasses",
    srcs = [
        "RunModel/AppDelegate.mm",
        "RunModel/RunModelViewController.mm",
	"RunModel/ios_image_load.mm",
    ],
    hdrs = glob(["RunModel/*.h"]),
    xibs = ["RunModel/RunModelViewController.xib"],
    deps = [
	"//google/protobuf:protobuf",
	"//google/protobuf:protobuf_lite",
	"//tensorflow/core:android_tensorflow_lib",
	"//third_party/eigen3:eigen3",
    ],
    copts = ["--std=c++11", "-Wl,--whole-archive", "-Wl,-all_load", "-femit-all-decls", "-O0"],
    includes = ["../../../external/eigen_archive"],
    sdk_frameworks = ["CoreGraphics"],
)

objc_binary(
    name = "ios-app-binary",
    srcs = [
        "RunModel/main.mm",
    ],
    deps = [
        ":RunModelClasses",
    ],
    copts = ["-Wl,--whole-archive", "-Wl,-all_load", "-femit-all-decls", "-O3"],
)

ios_application(
    name = "ios-app",
    binary = ":ios-app-binary",
    infoplist = "RunModel/RunModel-Info.plist",
)

objc_binary(
    name = "ios-app-binary-cc",
    srcs = [
        "RunModel/main.mm",
    ],
    deps = [
        ":RunModelClasses",
	"//google/protobuf:protobuf",
	"//google/protobuf:protobuf_lite",
    ],
    resources = [
        "data/grace_hopper.jpg",
        "data/imagenet_comp_graph_label_strings.txt",
        "data/tensorflow_inception_graph.pb",
    ],
    copts = ["-Wl,--whole-archive", "-femit-all-decls"],
)
