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

load("@flatbuffers//:build_defs.bzl", "flatbuffer_cc_library", "flatbuffer_java_library", "flatc_path")

package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

genrule(
    name = "configuration_schema",
    srcs = ["configuration.proto"],
    outs = ["configuration.fbs"],
    # We rename the namespace since otherwise the proto classes and flatbuffer
    # classes would have the same names.
    cmd = """
    $(location {}) --proto -o $(@D) $(location :configuration.proto)
    perl -p -i -e 's/tflite.proto/tflite/' $(@D)/configuration.fbs
    """.format(flatc_path),
    tools = [
        flatc_path,
    ],
)

genrule(
    name = "configuration_fbs_contents_cc",
    srcs = ["configuration.fbs"],
    outs = ["configuration_fbs_contents-inl.h"],
    cmd = """
      echo 'constexpr char configuration_fbs_contents[] = R"Delimiter(' > $(@)
      cat < $(<) >> $(@)
      echo ')Delimiter";' >> $(@)
    """,
)

proto_library(
    name = "configuration_proto",
    srcs = [
        "configuration.proto",
    ],
)

cc_proto_library(
    name = "configuration_cc_proto",
    deps = [":configuration_proto"],
)

java_lite_proto_library(
    name = "configuration_java_proto_lite",
    deps = [":configuration_proto"],
)

flatbuffer_cc_library(
    name = "configuration_fbs",
    srcs = [":configuration.fbs"],
)

flatbuffer_java_library(
    name = "configuration_fbs_java",
    srcs = [":configuration.fbs"],
)

cc_library(
    name = "proto_to_flatbuffer",
    srcs = [
        "configuration_fbs_contents-inl.h",
        "proto_to_flatbuffer.cc",
    ],
    hdrs = ["proto_to_flatbuffer.h"],
    deps = [
        ":configuration_cc_proto",
        ":configuration_fbs",
        "//tensorflow/core/platform:protobuf",
        "//tensorflow/lite:minimal_logging",
        "@flatbuffers",
    ],
)

cc_library(
    name = "delegate_registry",
    srcs = ["delegate_registry.cc"],
    hdrs = ["delegate_registry.h"],
    deps = [
        ":configuration_fbs",
        "//tensorflow/lite/c:common",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "nnapi_plugin",
    srcs = ["nnapi_plugin.cc"],
    deps = [
        ":configuration_fbs",
        ":delegate_registry",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,  # For registration to always run.
)

cc_test(
    name = "nnapi_plugin_test",
    srcs = ["nnapi_plugin_test.cc"],
    tags = [
        "no_mac",
        "no_windows",
        "tflite_not_portable_ios",
    ],
    deps = [
        ":configuration_fbs",
        ":delegate_registry",
        ":nnapi_plugin",
        "//tensorflow/lite:framework",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate",
        "//tensorflow/lite/delegates/nnapi:nnapi_delegate_mock_test",
        "//tensorflow/lite/kernels:test_util",
        "@com_google_googletest//:gtest_main",
        "@flatbuffers",
    ],
)

cc_library(
    name = "hexagon_plugin",
    srcs = ["hexagon_plugin.cc"],
    deps = [
        ":configuration_fbs",
        ":delegate_registry",
        "@com_google_absl//absl/memory",
    ] + select({
        "//tensorflow:android": [
            "//tensorflow/lite/delegates/hexagon:hexagon_delegate",
        ],
        "//conditions:default": [],
    }),
    alwayslink = 1,  # For registration to always run.
)

cc_library(
    name = "gpu_plugin",
    srcs = ["gpu_plugin.cc"],
    deps = [
        ":configuration_fbs",
        ":delegate_registry",
        "//tensorflow/lite/delegates/gpu:delegate",
        "@com_google_absl//absl/memory",
    ],
    alwayslink = 1,  # For registration to always run.
)
