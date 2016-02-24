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
    resources = [
        "data/grace_hopper.jpg",
        "data/imagenet_comp_graph_label_strings.txt",
        "data/tensorflow_inception_graph.pb",
    ],
)

config_setting(
    name = "ios_armv7",
    values = {
        "ios_cpu": "armv7",
    },
)

config_setting(
    name = "ios_armv7s",
    values = {
        "ios_cpu": "armv7s",
    },
)

config_setting(
    name = "ios_arm64",
    values = {
        "ios_cpu": "arm64",
    },
)

IOS_ARM_TF_COPTS = [
    "-DOS_IOS",
    "-miphoneos-version-min=7.0",
    "-arch armv7",
    "-arch armv7s",
    "-arch arm64",
    "-D__thread=",
    "-isysroot /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS9.2.sdk/",
]

cc_proto_library(
    name = "proto_smoke_test",
    srcs = ["UrlGet/test.proto"],
    copts = select({
        ":ios_armv7": IOS_ARM_TF_COPTS,
        ":ios_armv7s": IOS_ARM_TF_COPTS,
        ":ios_arm64": IOS_ARM_TF_COPTS,
        "//conditions:default": [],
    }),
)
