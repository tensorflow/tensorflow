load("//tensorflow:tensorflow.bzl", "get_compatible_with_portable")

package(
    licenses = ["notice"],
)

cc_library(
    name = "headers",
    hdrs = [
        "interpreter.h",
    ],
    compatible_with = get_compatible_with_portable(),
    visibility = [
        "//tensorflow/lite:__subpackages__",
    ],
    deps = [
        "//tensorflow/lite:allocation",
        "//tensorflow/lite:external_cpu_backend_context",
        "//tensorflow/lite:interpreter_options_header",
        "//tensorflow/lite:signature_runner_header",
        "//tensorflow/lite:stderr_reporter",
        "//tensorflow/lite:string",
        "//tensorflow/lite:subgraph",
        "//tensorflow/lite:type_to_tflitetype",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/core/api:error_reporter",
        "//tensorflow/lite/experimental/resource",
        "//tensorflow/lite/internal:signature_def",
        "//tensorflow/lite/profiling:root_profiler",
    ],
)
