# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

load("@rules_shell//shell:sh_test.bzl", "sh_test")
load("@xla//third_party/rules_python/python:py_binary.bzl", "py_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_binary(
    name = "debug_fibonacci",
    srcs = ["debug_fibonacci.py"],
    strict_deps = True,
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "debug_errors",
    srcs = ["debug_errors.py"],
    strict_deps = True,
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "debug_keras",
    srcs = ["debug_keras.py"],
    strict_deps = True,
    deps = [
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/debug:debug_py",
        "//third_party/py/numpy",
    ],
)

py_binary(
    name = "debug_mnist",
    srcs = ["debug_mnist_v1.py"],
    main = "debug_mnist_v1.py",
    strict_deps = True,
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
