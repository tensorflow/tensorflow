load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("//tensorflow:strict.default.bzl", "py_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_strict_binary(
    name = "debug_fibonacci",
    srcs = ["debug_fibonacci.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
    ],
)

py_strict_binary(
    name = "debug_errors",
    srcs = ["debug_errors.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
    ],
)

py_strict_binary(
    name = "debug_keras",
    srcs = ["debug_keras.py"],
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
    ],
)

py_strict_binary(
    name = "debug_mnist",
    srcs = ["debug_mnist_v1.py"],
    main = "debug_mnist_v1.py",
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
    ],
)

sh_test(
    name = "examples_v1_debug_errors_test",
    srcs = ["examples_v1_debug_errors_test.sh"],
    data = [
        ":debug_errors",
    ],
    tags = [
        "no_windows",
        "noasan",  # TODO(b/190625515)
        "v1only",
    ],
)

sh_test(
    name = "examples_v1_debug_fibonacci_test",
    srcs = ["examples_v1_debug_fibonacci_test.sh"],
    data = [
        ":debug_fibonacci",
    ],
    tags = [
        "no_windows",
        "v1only",
    ],
)

sh_test(
    name = "examples_v1_debug_keras_test",
    size = "medium",
    srcs = ["examples_v1_debug_keras_test.sh"],
    data = [
        ":debug_keras",
    ],
    tags = [
        "no_windows",
        "noasan",  # TODO(b/190625515)
        "v1only",
    ],
)

sh_test(
    name = "examples_v1_debug_mnist_test",
    srcs = ["examples_v1_debug_mnist_test.sh"],
    data = [
        ":debug_mnist",
    ],
    tags = [
        "no_windows",
        "noasan",  # TODO(b/193153560)
        "v1only",
    ],
)

sh_test(
    name = "examples_v1_offline_analyzer_test",
    srcs = ["examples_v1_offline_analyzer_test.sh"],
    data = [
        "//tensorflow/python/debug/cli:offline_analyzer",
    ],
    tags = [
        "no_windows",
        "v1only",
    ],
)
