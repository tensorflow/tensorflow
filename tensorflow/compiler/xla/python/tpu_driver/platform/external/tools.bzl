# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""
Build dependencies and utilities for the TPU driver interface.
"""

load(
    "//tensorflow/core/platform:default/build_config.bzl",
    _tf_proto_library_cc = "tf_proto_library_cc",
)

def tf_proto_library_cc(**kwargs):
    kwargs["use_grpc_namespace"] = True
    _tf_proto_library_cc(**kwargs)

def go_proto_library(**kwargs):
    # A dummy macro placeholder for compatibility reason.
    pass

def go_grpc_library(**kwargs):
    # A dummy macro placeholder for compatibility reason.
    pass

def external_deps():
    return [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "//tensorflow:grpc++",
    ]

def tpu_proto_deps():
    return [
        ":tpu_driver_proto",
    ]

def tpu_grpc_proto_deps():
    return [
        ":tpu_service_proto",
    ]
