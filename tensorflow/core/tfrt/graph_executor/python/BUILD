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

# This package loads pybind11 dependencies to ensure compatibility with Python API.
# Files affected: graph_execution_options
load("//tensorflow:tensorflow.default.bzl", "tf_python_pybind_extension")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":friends"],
    licenses = ["notice"],
)

package_group(
    name = "friends",
    packages = [
        # Authorized users go here.
        "//tensorflow/core/tfrt/saved_model/...",
        "//tensorflow/core/tfrt/graph_executor/...",
    ],
)

tf_python_pybind_extension(
    name = "_pywrap_graph_execution_options",
    srcs = ["graph_execution_options.cc"],
    enable_stub_generation = True,
    module_name = "_pywrap_graph_execution_options",
    pytype_srcs = [
        "_pywrap_graph_execution_options.pyi",
    ],
    deps = [
        "//tensorflow/core/tfrt/graph_executor",
        "//tensorflow/core/tfrt/graph_executor:graph_execution_options",
        "//tensorflow/core/tfrt/runtime",
        "//tensorflow/core/tfrt/saved_model",
        "//tensorflow/core/tfrt/saved_model:saved_model_aot_compile",
        "@pybind11",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@xla//third_party/python_runtime:headers",
    ],
)
