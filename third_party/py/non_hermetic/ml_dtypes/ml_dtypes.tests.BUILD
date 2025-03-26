package(
    default_visibility = ["//visibility:public"],
)

py_library(
    name = "testing_base",
    deps = [
        "//:ml_dtypes",
        "@absl_py//absl/testing:absltest",
        "@absl_py//absl/testing:parameterized",
        "@local_xla//third_party/py/numpy",
    ],
)

py_test(
    name = "custom_float_test",
    srcs = ["custom_float_test.py"],
    main = "custom_float_test.py",
    deps = [":testing_base"],
)

py_test(
    name = "int4_test",
    srcs = ["int4_test.py"],
    main = "int4_test.py",
    deps = [":testing_base"],
)

py_test(
    name = "iinfo_test",
    srcs = ["iinfo_test.py"],
    main = "iinfo_test.py",
    deps = [":testing_base"],
)

py_test(
    name = "finfo_test",
    srcs = ["finfo_test.py"],
    main = "finfo_test.py",
    deps = [":testing_base"],
)

py_test(
    name = "metadata_test",
    srcs = ["metadata_test.py"],
    main = "metadata_test.py",
    deps = [":testing_base"],
)

cc_test(
    name = "float8_test",
    srcs = ["float8_test.cc"],
    linkstatic = 1,
    deps = [
        "//:float8",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)

cc_test(
    name = "intn_test_cc",
    srcs = ["intn_test.cc"],
    linkstatic = 1,
    deps = [
        "//:intn",
        "@com_google_googletest//:gtest_main",
        "@eigen_archive//:eigen3",
    ],
)
