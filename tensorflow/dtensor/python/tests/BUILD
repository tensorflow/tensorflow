load("//tensorflow:pytype.default.bzl", "pytype_library")
load(
    "//tensorflow/dtensor:build_defs.bzl",
    "dtensor_test",
)

pytype_library(
    name = "test_util",
    testonly = False,  # build_pip_package depends on this target.
    srcs = [
        "test_backend_name.py",
        "test_util.py",
    ],
    visibility = [
        "//tensorflow/dtensor:dtensor-internal",
        "//tensorflow/dtensor:dtensor-users",
        "//tensorflow/tools/pip_package:__pkg__",
    ],
    deps = [
        "//tensorflow/dtensor/python:api",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/dtensor/python:layout",
        "//tensorflow/dtensor/python:tpu_util",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:config",
        "//tensorflow/python/platform:client_testlib",
        "//third_party/py/numpy",
        "@absl_py//absl/flags",
        "@absl_py//absl/testing:parameterized",
    ],
)

dtensor_test(
    name = "config_test",
    srcs = ["config_test.py"],
    main = "config_test.py",
    deps = [
        ":test_util",
        "//tensorflow/dtensor/python:config",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/framework:for_generated_wrappers",
    ],
)
