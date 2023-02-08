load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/tools/evaluation/tasks:build_def.bzl", "task_linkopts")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "dummy_delegate",
    srcs = [
        "dummy_delegate.cc",
    ],
    hdrs = [
        "dummy_delegate.h",
    ],
    deps = [
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/utils:simple_delegate",
    ],
)

cc_binary(
    name = "dummy_external_delegate.so",
    srcs = [
        "external_delegate_adaptor.cc",
    ],
    defines = ["TFL_EXTERNAL_DELEGATE_COMPILE_LIBRARY"],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":dummy_delegate",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/delegates/external:external_delegate_interface",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
    ],
)

#### The following are for using the dummy test delegate in TFLite tooling ####
cc_library(
    name = "dummy_delegate_provider",
    srcs = ["dummy_delegate_provider.cc"],
    copts = tflite_copts(),
    deps = [
        ":dummy_delegate",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "benchmark_model_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/benchmark:benchmark_model_main",
    ],
)

cc_binary(
    name = "inference_diff_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/inference_diff:run_eval_lib",
    ],
)

cc_binary(
    name = "imagenet_classification_eval_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval_lib",
    ],
)

cc_binary(
    name = "coco_object_detection_eval_plus_dummy_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":dummy_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval_lib",
    ],
)

sh_test(
    name = "external_delegate_test",
    srcs = ["external_delegate_test.sh"],
    data = [
        "//tensorflow/lite/delegates/coreml/internal_test/testdata:mobilenet_v2_1.0_224_quantized_weights_fp16.tflite",
        "//tensorflow/lite/delegates/utils/dummy_delegate:dummy_external_delegate.so",
        "//tensorflow/lite/tools/benchmark:benchmark_model",
    ],
    visibility = ["//visibility:private"],
)
