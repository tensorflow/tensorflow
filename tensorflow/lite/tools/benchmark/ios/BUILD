load(
    "@build_bazel_rules_apple//apple:ios.bzl",
    "ios_application",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

objc_library(
    name = "benchmark_lib",
    srcs = [
        "TFLiteBenchmark/TFLiteBenchmark/AppDelegate.m",
        "TFLiteBenchmark/TFLiteBenchmark/BenchmarkViewController.mm",
        "TFLiteBenchmark/TFLiteBenchmark/main.m",
    ],
    hdrs = [
        "TFLiteBenchmark/TFLiteBenchmark/AppDelegate.h",
        "TFLiteBenchmark/TFLiteBenchmark/BenchmarkViewController.h",
    ],
    copts = [
        "-DUSE_TFLITE_BENCHMARK_HEADERS",
    ],
    deps = [
        "//tensorflow/lite/tools/benchmark/experimental/c:benchmark_c_api",
    ],
)

# Add model files to this rule to get packaged with the Application for benchmarking.
filegroup(
    name = "models",
    srcs = [
    ],
)

ios_application(
    name = "benchmark_app",
    app_icons = glob(["TFLiteBenchmark/TFLiteBenchmark/Assets.xcassets/AppIcon.appiconset/*"]),
    bundle_id = "example.TFLiteBenchmark",
    families = [
        "iphone",
    ],
    infoplists = ["TFLiteBenchmark/TFLiteBenchmark/Info.plist"],
    launch_storyboard = "TFLiteBenchmark/TFLiteBenchmark/Base.lproj/Main.storyboard",
    minimum_os_version = "11.0",
    provisioning_profile = "//googlemac/iPhone/Shared/ProvisioningProfiles:Google_Development.mobileprovision",
    resources = [
        "TFLiteBenchmark/TFLiteBenchmark/benchmark_data/benchmark_params.json",
        ":models",
    ],
    deps = [
        ":benchmark_lib",
    ],
)
