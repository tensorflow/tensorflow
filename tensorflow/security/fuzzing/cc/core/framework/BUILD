# Fuzztest TensorShape domains

cc_library(
    name = "tensor_shape_domains",
    testonly = 1,
    srcs = ["tensor_shape_domains.cc"],
    hdrs = ["tensor_shape_domains.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/platform:statusor",
        "@com_google_fuzztest//fuzztest",
    ],
)

cc_library(
    name = "tensor_domains",
    testonly = 1,
    srcs = ["tensor_domains.cc"],
    hdrs = ["tensor_domains.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":datatype_domains",
        ":tensor_shape_domains",
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/platform:statusor",
        "@com_google_fuzztest//fuzztest",
    ],
)

cc_library(
    name = "datatype_domains",
    testonly = 1,
    srcs = ["datatype_domains.cc"],
    hdrs = ["datatype_domains.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core/framework:types_proto_cc",
        "@com_google_fuzztest//fuzztest",
    ],
)
