# TensorFlow Lite for Swift.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application", "ios_unit_test")
load("@build_bazel_rules_swift//swift:swift.bzl", "swift_library")

MINIMUM_OS_VERSION = "9.0"

SWIFT_COPTS = [
    "-wmo",
]

swift_library(
    name = "TensorFlowLite",
    srcs = glob(["Sources/*.swift"]),
    copts = SWIFT_COPTS,
    module_name = "TensorFlowLite",
    tags = ["manual"],
    deps = [
        "//tensorflow/lite/experimental/c:c_api",
    ],
)

ios_unit_test(
    name = "TensorFlowLiteTests",
    size = "small",
    minimum_os_version = MINIMUM_OS_VERSION,
    tags = [
        "manual",
        # DISABLED: Following sanitizer tests are not supported by iOS test targets.
        "noasan",
        "nomsan",
        "notsan",
    ],
    deps = [":TensorFlowLiteTestsLib"],
)

swift_library(
    name = "TensorFlowLiteTestsLib",
    testonly = 1,
    srcs = glob(["Tests/*.swift"]),
    copts = SWIFT_COPTS,
    tags = ["manual"],
    deps = [
        ":TensorFlowLite",
        ":TestResources",
    ],
)

objc_library(
    name = "TestResources",
    resources = [
        "//tensorflow/lite:testdata/add.bin",
        "//tensorflow/lite:testdata/add_quantized.bin",
        "//tensorflow/lite:testdata/multi_add.bin",
    ],
    tags = ["manual"],
)

ios_application(
    name = "TensorFlowLiteApp",
    app_icons = glob(["TestApps/TensorFlowLiteApp/TensorFlowLiteApp/Assets.xcassets/AppIcon.appiconset/**"]),
    bundle_id = "com.tensorflow.lite.swift.TensorFlowLite",
    families = [
        "ipad",
        "iphone",
    ],
    infoplists = ["TestApps/TensorFlowLiteApp/TensorFlowLiteApp/Info.plist"],
    launch_storyboard = "TestApps/TensorFlowLiteApp/TensorFlowLiteApp/Base.lproj/LaunchScreen.storyboard",
    minimum_os_version = MINIMUM_OS_VERSION,
    sdk_frameworks = [
        "CoreGraphics",
    ],
    tags = ["manual"],
    deps = [":TensorFlowLiteAppLib"],
)

swift_library(
    name = "TensorFlowLiteAppLib",
    srcs = glob(["TestApps/TensorFlowLiteApp/TensorFlowLiteApp/*.swift"]),
    module_name = "TensorFlowLiteAppLib",
    tags = ["manual"],
    deps = [
        ":TensorFlowLite",
        ":TensorFlowLiteAppResources",
    ],
)

objc_library(
    name = "TensorFlowLiteAppResources",
    storyboards = glob([
        "TestApps/TensorFlowLiteApp/TensorFlowLiteApp/Base.lproj/*.storyboard",
    ]),
    tags = ["manual"],
    deps = [":TestResources"],
)
