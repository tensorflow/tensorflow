package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "generator",
    srcs = ["generator.cc"],
    hdrs = ["generator.h"],
    deps = [
        "//tensorflow/contrib/lite/schema:schema_fbs",
    ],
)

cc_binary(
    name = "generate",
    srcs = ["generate.cc"],
    deps = [
        ":generator",
    ],
)

cc_test(
    name = "generator_test",
    srcs = ["generator_test.cc"],
    tags = ["no_oss"],
    deps = [
        ":generator",
        "@com_google_googletest//:gtest",
    ],
)

cc_test(
    name = "consistency_test",
    srcs = ["consistency_test.cc"],
    data = [
        "//tensorflow/contrib/lite:builtin_ops.h",
    ],
    tags = ["no_oss"],
    deps = [
        ":generator",
        "@com_google_googletest//:gtest",
    ],
)
