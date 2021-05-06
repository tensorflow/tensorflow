# Description:
#   TensorFlow Lite for Microcontrollers Vision Example.
package(
    default_visibility = ["//visibility:public"],
    features = ["-layering_check"],
    licenses = ["notice"],
)

cc_library(
    name = "model_settings",
    srcs = [
        "model_settings.cc",
    ],
    hdrs = [
        "model_settings.h",
    ],
)

cc_library(
    name = "person_detect_model_data",
    srcs = [
        "@person_detect_data//:person_detect_model_data",
    ],
    hdrs = [
        "person_detect_model_data.h",
    ],
    tags = [
        "no_oss",  # TODO(b/174680668): Exclude from OSS.
    ],
)

cc_library(
    name = "simple_images_test_data",
    srcs = [
        "no_person_image_data.cc",
        "person_image_data.cc",
    ],
    hdrs = [
        "no_person_image_data.h",
        "person_image_data.h",
    ],
    tags = [
        "no_oss",  # TODO(b/174680668): Exclude from OSS.
    ],
    deps = [
        ":model_settings",
    ],
)

cc_test(
    name = "person_detection_test",
    srcs = ["person_detection_test.cc"],
    tags = [
        "no_oss",  # TODO(b/174680668): Exclude from OSS.
    ],
    deps = [
        ":model_settings",
        ":person_detect_model_data",
        ":simple_images_test_data",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro/testing:micro_test",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)

cc_library(
    name = "image_provider",
    srcs = [
        "image_provider.cc",
    ],
    hdrs = [
        "image_provider.h",
    ],
    deps = [
        ":model_settings",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_error_reporter",
    ],
)

cc_test(
    name = "image_provider_test",
    srcs = [
        "image_provider_test.cc",
    ],
    deps = [
        ":image_provider",
        ":model_settings",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_library(
    name = "detection_responder",
    srcs = [
        "detection_responder.cc",
    ],
    hdrs = [
        "detection_responder.h",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/micro:micro_error_reporter",
    ],
)

cc_test(
    name = "detection_responder_test",
    srcs = [
        "detection_responder_test.cc",
    ],
    deps = [
        ":detection_responder",
        "//tensorflow/lite/micro/testing:micro_test",
    ],
)

cc_binary(
    name = "person_detection",
    srcs = [
        "main.cc",
        "main_functions.cc",
        "main_functions.h",
    ],
    tags = [
        "no_oss",  # TODO(b/174680668): Exclude from OSS.
    ],
    deps = [
        ":detection_responder",
        ":image_provider",
        ":model_settings",
        ":person_detect_model_data",
        "//tensorflow/lite/micro:micro_error_reporter",
        "//tensorflow/lite/micro:micro_framework",
        "//tensorflow/lite/micro:op_resolvers",
        "//tensorflow/lite/micro:system_setup",
        "//tensorflow/lite/schema:schema_fbs",
    ],
)
