# Fuzztest TensorShape domains

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

cc_library(
    name = "tensor_shape_domains",
    testonly = 1,
    srcs = ["tensor_shape_domains.cc"],
    hdrs = ["tensor_shape_domains.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "@com_google_absl//absl/status:statusor",
        "@com_google_fuzztest//fuzztest",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "tensor_domains",
    testonly = 1,
    srcs = ["tensor_domains.cc"],
    hdrs = ["tensor_domains.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":tensor_shape_domains",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:tstring",
        "@com_google_absl//absl/log",
        "@com_google_fuzztest//fuzztest",
        "@local_xla//xla/tsl/platform:errors",
    ],
)

cc_library(
    name = "datatype_domains",
    testonly = 1,
    srcs = ["datatype_domains.cc"],
    hdrs = ["datatype_domains.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_fuzztest//fuzztest",
    ],
)
