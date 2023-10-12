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

package(default_visibility = [
    "//visibility:public",
])

licenses([
    "notice",  # BSD-3-Clause-Clear
])

exports_files(glob(["hexagon/**/*.so"]))

#Just header file, needed for data types in the interface.
cc_library(
    name = "hexagon_nn_header",
    hdrs = [
        "@hexagon_nn//:hexagon/hexagon_nn.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
)

cc_library(
    name = "hexagon_nn_ops",
    hdrs = [
        "@hexagon_nn//:hexagon/hexagon_nn_ops.h",
        "@hexagon_nn//:hexagon/ops.def",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
)

cc_library(
    name = "remote",
    hdrs = [
        "@hexagon_nn//:hexagon/remote.h",
        "@hexagon_nn//:hexagon/remote64.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
)

cc_library(
    name = "rpcmem",
    srcs = [
        "@hexagon_nn//:hexagon/rpcmem_stub.c",
    ],
    hdrs = [
        "@hexagon_nn//:hexagon/rpcmem.h",
    ],
    deps = [
        ":AEEStdDef",
    ],
)

cc_library(
    name = "hexagon_soc",
    hdrs = [
        "@hexagon_nn//:hexagon/hexnn_soc_defines.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
)

cc_library(
    name = "AEEStdDef",
    hdrs = [
        "@hexagon_nn//:hexagon/AEEStdDef.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
)

cc_library(
    name = "AEEStdErr",
    hdrs = [
        "@hexagon_nn//:hexagon/AEEStdErr.h",
    ],
)

# The files are included in another .c files, so we add the src files as textual_hdrs
# to avoid compiling them and have linking errors for multiple symbols.
cc_library(
    name = "hexagon_stub",
    textual_hdrs = [
        "@hexagon_nn//:hexagon/hexagon_nn_domains_stub.c",
        "@hexagon_nn//:hexagon/hexagon_nn_stub.c",
    ],
)

# This rule uses the smart/graph wrapper interfaces.
cc_library(
    name = "hexagon_nn",
    srcs = [
        "@hexagon_nn//:hexagon/hexagon_nn.h",
        "@hexagon_nn//:hexagon/hexnn_dsp_api.h",
        "@hexagon_nn//:hexagon/hexnn_dsp_api_impl.c",
        "@hexagon_nn//:hexagon/hexnn_dsp_domains_api.h",
        "@hexagon_nn//:hexagon/hexnn_dsp_domains_api_impl.c",
        "@hexagon_nn//:hexagon/hexnn_dsp_smart_wrapper_api.c",
        "@hexagon_nn//:hexagon/hexnn_dsp_smart_wrapper_api.h",
        "@hexagon_nn//:hexagon/hexnn_graph_wrapper.cpp",
        "@hexagon_nn//:hexagon/hexnn_graph_wrapper.hpp",
        "@hexagon_nn//:hexagon/hexnn_graph_wrapper_interface.h",
        "@hexagon_nn//:hexagon/hexnn_soc_defines.h",
    ],
    deps = [
        ":AEEStdErr",
        ":hexagon_stub",
        ":remote",
        ":rpcmem",
    ],
    alwayslink = 1,
)
