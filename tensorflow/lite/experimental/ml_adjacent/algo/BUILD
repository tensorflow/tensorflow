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
