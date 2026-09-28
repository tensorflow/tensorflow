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

# TensorFlow API backwards compatibility tests.

load(
    "//tensorflow:tensorflow.bzl",
    "py_test",
    "tf_cc_binary",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/tools/api:__subpackages__"],
    licenses = ["notice"],
)

exports_files([
    "README.txt",
    "API_UPDATE_WARNING.txt",
])

py_test(
    name = "api_compatibility_test",
    srcs = ["api_compatibility_test.py"],
    data = [
        ":API_UPDATE_WARNING.txt",
        ":README.txt",
        "//tensorflow/tools/api/golden:api_golden_v1",
        "//tensorflow/tools/api/golden:api_golden_v2",
        "//third_party/py/numpy/tf_numpy_api:api_golden",
    ],
    strict_deps = True,
    tags = [
        "no_mac",  # b/198669105
        "no_oss",  # Runs explicitly in OSS
        "no_pip",
        "no_rocm",
        "no_windows",  # Bugs due to some paths.
    ],
    deps = [
        # copybara:uncomment #"//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python/lib/io:file_io",
        "//tensorflow/python/platform:client_testlib",
        "//tensorflow/python/platform:resource_loader",
        "//tensorflow/python/platform:tf_logging",
        "//tensorflow/tools/api/lib:api_objects_proto_py",
        "//tensorflow/tools/api/lib:python_object_to_proto_visitor",
        "//tensorflow/tools/common:public_api",
        "//tensorflow/tools/common:traverse",
    ],
)

py_test(
    name = "module_test",
    srcs = ["module_test.py"],
    strict_deps = True,
    tags = [
        "no_windows",  # Failing due to missing API symbols.
    ],
    deps = [
        # copybara:uncomment "//third_party/py/google/protobuf:use_fast_cpp_protos",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:tf2",
        "//tensorflow/python/platform:client_testlib",
    ],
)

tf_cc_binary(
    name = "convert_from_multiline",
    srcs = ["convert_from_multiline.cc"],
    deps = [
        "//tensorflow/core:lib",
        "//tensorflow/core:op_gen_lib",
        "@com_google_absl//absl/status",
    ],
)
