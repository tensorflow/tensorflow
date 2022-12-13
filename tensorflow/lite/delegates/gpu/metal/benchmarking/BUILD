load("@build_bazel_rules_apple//apple:ios.bzl", "ios_application")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

objc_library(
    name = "benchmark_lib",
    srcs = ["main.mm"],
    data = glob([
        "models/*.tflite",
    ]),
    sdk_frameworks = [
        "Metal",
        "QuartzCore",
    ],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/gpu/common:gpu_info",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:model_builder",
        "//tensorflow/lite/delegates/gpu/common:model_transformer",
        "//tensorflow/lite/delegates/gpu/common:precision",
        "//tensorflow/lite/delegates/gpu/common:shape",
        "//tensorflow/lite/delegates/gpu/common:util",
        "//tensorflow/lite/delegates/gpu/common/transformations:model_transformations",
        "//tensorflow/lite/delegates/gpu/metal:inference_context",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:kernel_util",
    ],
)

# Application for local benchmarking in XCode
ios_application(
    name = "iOSBenchmark",
    testonly = 1,
    bundle_id = "com.tensorflow.lite.delegates.gpu.metal.benchmarking",
    families = [
        "iphone",
        "ipad",
    ],
    infoplists = ["Info.plist"],
    minimum_os_version = "12.0",
    provisioning_profile = "//tensorflow/lite/delegates/gpu/metal/benchmarking:provisioning_profile.mobileprovision",
    tags = [
        "local",
        "notap",
    ],
    deps = [":benchmark_lib"],
)
