# Copyright 2020 The TensorFlow Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for configuring googleapis in workspace external dependency.
"""

load("@com_google_googleapis//:repository_rules.bzl", "switched_rules_by_language")

def config_googleapis():
    """Configures external dependency googleapis to use Google's C++ gRPC APIs

    To avoid ODR violation, the cc proto libraries (*.pb.cc) must be
    statically linked to libtensorflow_framework.so, whereas cc grpc libraries
    (*.grpc.pb.cc) must be statically linked to _pywrap_tensorflow.so.

    To achieve this, Bazel rules are overridden to
      (1) generate headers-only proto library, and
      (2) build grpc library depending on the cc headers instead of the cc proto
          library.
    """
    switched_rules_by_language(
        name = "com_google_googleapis_imports",
        cc = True,
        grpc = True,
        rules_override = {
            "cc_proto_library": [
                "@org_tensorflow//third_party/googleapis:build_rules.bzl",
                "",
            ],
            "cc_grpc_library": [
                "@org_tensorflow//third_party/googleapis:build_rules.bzl",
                "",
            ],
        },
    )
