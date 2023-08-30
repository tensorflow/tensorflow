package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

cc_library(
    name = "crop",
    srcs = ["crop.cc"],
    hdrs = ["crop.h"],
    deps = [
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "crop_test",
    srcs = ["crop_test.cc"],
    deps = [
        ":crop",
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/experimental/ml_adjacent/data:owning_vector_ref",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "resize",
    srcs = ["resize.cc"],
    hdrs = ["resize.h"],
    deps = [
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "resize_test",
    srcs = ["resize_test.cc"],
    deps = [
        ":resize",
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/experimental/ml_adjacent/data:owning_vector_ref",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "rotate",
    srcs = ["rotate.cc"],
    hdrs = ["rotate.h"],
    deps = [
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "rotate_test",
    srcs = ["rotate_test.cc"],
    deps = [
        ":rotate",
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/experimental/ml_adjacent/data:owning_vector_ref",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "flip_left_right",
    srcs = ["flip_left_right.cc"],
    hdrs = ["flip_left_right.h"],
    deps = [
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "flip_left_right_test",
    srcs = ["flip_left_right_test.cc"],
    deps = [
        ":flip_left_right",
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/experimental/ml_adjacent/data:owning_vector_ref",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "flip_up_down",
    srcs = ["flip_up_down.cc"],
    hdrs = ["flip_up_down.h"],
    deps = [
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/kernels/internal:compatibility",
    ],
)

cc_test(
    name = "flip_up_down_test",
    srcs = ["flip_up_down_test.cc"],
    deps = [
        ":flip_up_down",
        "//tensorflow/lite/experimental/ml_adjacent:lib",
        "//tensorflow/lite/experimental/ml_adjacent/data:owning_vector_ref",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)
