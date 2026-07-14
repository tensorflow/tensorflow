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

load("//tensorflow/lite:build_def.bzl", "tflite_cc_shared_object")

package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

#Just header file, needed for data types in the interface.
cc_library(
    name = "hexagon_nn_header",
    hdrs = [
        "hexagon_nn.h",
        "hexagon_nn_init.h",
    ],
    deps = [
        "@hexagon_nn//:hexagon_nn_header",
    ],
)

# Fetches the QC SoC of the device.
cc_library(
    name = "soc_model",
    srcs = [
        "soc_model.cc",
    ],
    hdrs = [
        "soc_model.h",
    ],
    deps = [
        "@hexagon_nn//:hexagon_soc",
        "@hexagon_nn//:remote",
    ],
)

# DSP interface used to communicate to the aDSP/cDSP.
cc_library(
    name = "adsprpc_interface",
    srcs = ["adsprpc_interface.cc"],
    deps = [
        ":hexagon_nn_header",
        ":soc_model",
    ],
)

cc_library(
    name = "hexagon_nn_init",
    srcs = ["hexagon_nn_init.cc"],
    hdrs = ["hexagon_nn_init.h"],
    deps = [
        ":adsprpc_interface",
        ":soc_model",
        "@hexagon_nn//:AEEStdDef",
        "@hexagon_nn//:remote",
        "@hexagon_nn//:rpcmem",
    ],
    alwayslink = 1,
)

tflite_cc_shared_object(
    name = "libhexagon_interface.so",
    linkopts = [
        "-Wl,-soname=libhexagon_interface.so",
        "-z defs",
        "-Wl,--version-script,$(location //tensorflow/lite/delegates/hexagon/hexagon_nn:version_scripts.lds)",
    ] + select({
        "//tensorflow:android": [
            "-llog",
            "-ldl",
        ],
        "//conditions:default": [
            "-ldl",
        ],
    }),
    linkstatic = 1,
    deps = [
        ":hexagon_nn_init",
        ":version_scripts.lds",
        "@hexagon_nn",
    ],
)

cc_library(
    name = "hexagon_interface_so",
    srcs = [":libhexagon_interface.so"],
)

android_library(
    name = "hexagon_interface_android",
    exports = [":hexagon_interface_so"],
)

exports_files(["version_script.lds"])
