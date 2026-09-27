# Copyright 2026 Google LLC. All Rights Reserved.
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

# Fuzzing TensorFlow/core/function with GFT

load(
    "//tensorflow/security/fuzzing:tf_fuzzing.bzl",
    "tf_cc_fuzz_test",
)

# copybara:uncomment package(default_applicable_licenses = ["//tensorflow:license"])

tf_cc_fuzz_test(
    name = "runtime_client_fuzz",
    srcs = ["runtime_client_fuzz.cc"],
    deps = [
        "//tensorflow/c/eager:immediate_execution_context",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_types_hdr",
        "//tensorflow/core:portable_gif_internal",
        "//tensorflow/core/common_runtime/eager:context",
        "//tensorflow/core/framework:function_proto_cc",
        "//tensorflow/core/framework:tensor",
        "//tensorflow/core/framework:types_proto_cc",
        "//tensorflow/core/function/runtime_client:runtime_client_cc",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/strings",
        "@xla//xla/tsl/platform:status",
    ],
)
