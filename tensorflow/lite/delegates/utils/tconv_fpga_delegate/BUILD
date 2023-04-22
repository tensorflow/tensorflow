load("//tensorflow/lite:build_def.bzl", "tflite_copts")
load("//tensorflow/lite/tools/evaluation/tasks:build_def.bzl", "task_linkopts")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

cc_library(
    name = "tconv_fpga_delegate",
    srcs = [
        "tconv_fpga_delegate.cc",
    ],
    hdrs = [
        "tconv_fpga_delegate.h",
        "util.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils:simple_delegate",
        "//tensorflow/lite/kernels:padding",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/kernels/internal:types",
        "//tensorflow/lite/kernels/internal:tensor",
        "//tensorflow/lite/kernels/internal:common",
        "//tensorflow/lite/kernels/internal:optimized_base",
        "//tensorflow/lite/delegates/utils/tconv_fpga_delegate/accelerator/driver:driver",
    ],
)

cc_binary(
    name = "tconvfpga_external_delegate.so",
    srcs = [
        "external_delegate_adaptor.cc",
    ],
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":tconv_fpga_delegate",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
    ],
)

#### The following are for using the tconvfpga test delegate in TFLite tooling ####
cc_library(
    name = "tconv_fpga_delegate_provider",
    srcs = ["tconv_fpga_delegate_provider.cc"],
    copts = tflite_copts(),
    deps = [
        ":tconv_fpga_delegate",
        "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
    ],
    alwayslink = 1,
)



cc_binary(
    name = "label_image_plus_tconv_fpga_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/examples/label_image:label_image_lib",
        ":tconv_fpga_delegate_provider", 
    ],
)

cc_binary(
    name = "benchmark_model_plus_tconv_fpga_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        "//tensorflow/lite/tools/benchmark:benchmark_model_main",
        ":tconv_fpga_delegate_provider",
    ],
)

cc_binary(
    name = "inference_diff_plus_tconv_fpga_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":tconv_fpga_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/inference_diff:run_eval_lib",
    ],
)

cc_binary(
    name = "imagenet_classification_eval_plus_tconv_fpga_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":tconv_fpga_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification:run_eval_lib",
    ],
)

cc_binary(
    name = "coco_object_detection_eval_plus_tconv_fpga_delegate",
    copts = tflite_copts(),
    linkopts = task_linkopts(),
    deps = [
        ":tconv_fpga_delegate_provider",
        "//tensorflow/lite/tools/evaluation/tasks:task_executor_main",
        "//tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval_lib",
    ],
)
