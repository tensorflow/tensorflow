# Description:
#   Tensorflow camera demo app for Android.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//google/protobuf:protobuf.bzl", "cc_proto_library")
load("//google/protobuf:protobuf.bzl", "py_proto_library")

objc_library(
    name = "UrlGetClasses",
    srcs = [
        "UrlGet/AppDelegate.mm",
        "UrlGet/UrlGetViewController.mm",
    ],
    hdrs = glob(["UrlGet/*.h"]),
    xibs = ["UrlGet/UrlGetViewController.xib"],
    deps = [
	":proto_smoke_test",
	"//google/protobuf:protobuf",
	"//google/protobuf:protobuf_lite",
	"//tensorflow/core:android_tensorflow_lib",
	"//third_party/eigen3:eigen3",
    ],
    copts = ["--std=c++11"],
    includes = ["../../../external/eigen_archive"],
)

objc_binary(
    name = "ios-app-binary",
    srcs = [
        "UrlGet/main.m",
    ],
    deps = [
        ":UrlGetClasses",
    ],
)

ios_application(
    name = "ios-app",
    binary = ":ios-app-binary",
    infoplist = "UrlGet/UrlGet-Info.plist",
)

objc_binary(
    name = "ios-app-binary-cc",
    srcs = [
        "UrlGet/main.m",
    ],
    deps = [
        ":UrlGetClasses",
	"//google/protobuf:protobuf",
	"//google/protobuf:protobuf_lite",
    ],
)

cc_proto_library(
    name = "proto_smoke_test",
    srcs = ["UrlGet/test.proto"],
)
