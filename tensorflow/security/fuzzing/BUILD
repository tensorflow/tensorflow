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

# Fuzzing TensorFlow.
# Since we use OSSFuzz, gather all fuzzers into a single place.
# This way, we can use tooling to determine the status of the fuzzing efforts.

load("@xla//third_party/rules_python/python:py_library.bzl", "py_library")
load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_py_fuzz_target",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

py_library(
    name = "python_fuzzing",
    srcs = ["python_fuzzing.py"],
    strict_deps = True,
    deps = ["//tensorflow:tensorflow_py_no_contrib"],
)

tf_py_fuzz_target(
    name = "abs_fuzz",
    srcs = ["abs_fuzz.py"],
    tags = [
        "noautofuzz",
        "notap",
    ],  # Run in OSS only.
    deps = [":python_fuzzing"],
)

tf_py_fuzz_target(
    name = "acos_fuzz",
    srcs = ["acos_fuzz.py"],
    tags = [
        "noautofuzz",
        "notap",
    ],  # Run in OSS only.
    deps = [":python_fuzzing"],
)

tf_py_fuzz_target(
    name = "acosh_fuzz",
    srcs = ["acosh_fuzz.py"],
    tags = [
        "noautofuzz",
        "notap",
    ],  # Run in OSS only.
    deps = [":python_fuzzing"],
)

tf_py_fuzz_target(
    name = "add_fuzz",
    srcs = ["add_fuzz.py"],
    tags = [
        "noautofuzz",
        "notap",
    ],  # Run in OSS only.
    deps = [":python_fuzzing"],
)

tf_py_fuzz_target(
    name = "constant_fuzz",
    srcs = ["constant_fuzz.py"],
    tags = [
        "noautofuzz",
        "notap",
    ],  # Run in OSS only.
)

tf_py_fuzz_target(
    name = "raggedCountSparseOutput_fuzz",
    srcs = ["raggedCountSparseOutput_fuzz.py"],
    tags = [
        "noautofuzz",
        "notap",
    ],  # Run in OSS only.
    deps = [":python_fuzzing"],
)

tf_py_fuzz_target(
    name = "dataFormatVecPermute_fuzz",
    srcs = ["dataFormatVecPermute_fuzz.py"],
    tags = [
        "noautofuzz",
        "notap",
    ],  # Run in OSS only.
    deps = [":python_fuzzing"],
)

tf_py_fuzz_target(
    name = "sparseCountSparseOutput_fuzz",
    srcs = ["sparseCountSparseOutput_fuzz.py"],
    tags = [
        "noautofuzz",
        "notap",
    ],  # Run in OSS only.
    deps = [":python_fuzzing"],
)

tf_py_fuzz_target(
    name = "tf2migration_fuzz",
    srcs = ["tf2migration_fuzz.py"],
    tags = [
        "noautofuzz",
        "notap",
    ],  # Run in OSS only.
    deps = [":python_fuzzing"],
)
