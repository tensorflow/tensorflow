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
