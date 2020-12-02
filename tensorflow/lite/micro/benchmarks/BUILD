licenses(["notice"])  # Apache 2.0

package_group(
    name = "micro_top_level",
    packages = ["//tensorflow/lite/micro"],
)

cc_library(
    name = "micro_benchmark",
    hdrs = [
        "micro_benchmark.h",
    ],
    deps = [
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_time",
        "//tensorflow/lite/micro:op_resolvers",
    ],
)

cc_library(
    name = "keyword_scrambled_model_data",
    srcs = [
        "keyword_scrambled_model_data.cc",
    ],
    hdrs = [
        "keyword_scrambled_model_data.h",
    ],
    visibility = [
        ":micro_top_level",
    ],
)

cc_binary(
    name = "keyword_benchmark",
    srcs = ["keyword_benchmark.cc"],
    deps = [
        ":keyword_scrambled_model_data",
        ":micro_benchmark",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/kernels:fully_connected",
    ],
)

cc_binary(
    name = "person_detection_benchmark",
    srcs = ["person_detection_benchmark.cc"],
    tags = [
        "no_oss",  # TODO(b/174680668): Exclude from OSS.
    ],
    deps = [
        ":micro_benchmark",
        "//tensorflow/lite:schema_fbs_version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/examples/person_detection:model_settings",
        "//tensorflow/lite/micro/examples/person_detection:person_detect_model_data",
        "//tensorflow/lite/micro/examples/person_detection:simple_images_test_data",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "person_detection_experimental_benchmark",
    srcs = ["person_detection_experimental_benchmark.cc"],
    tags = [
        "no_oss",  # TODO(b/174680668): Exclude from OSS.
    ],
    deps = [
        ":micro_benchmark",
        "//tensorflow/lite:version",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:micro_utils",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/examples/person_detection_experimental:model_settings",
        "//tensorflow/lite/micro/examples/person_detection_experimental:person_detect_model_data",
        "//tensorflow/lite/micro/examples/person_detection_experimental:simple_images_test_data",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
