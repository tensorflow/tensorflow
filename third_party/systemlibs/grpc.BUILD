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

licenses(["notice"])  # Apache v2

filegroup(
    name = "LICENSE",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "grpc",
    linkopts = [
        "-lgrpc",
        "-lgpr",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "grpc++",
    linkopts = [
        "-lgrpc++",
        "-lgpr",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "grpc++_codegen_proto",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "grpc_unsecure",
    linkopts = [
        "-lgrpc_unsecure",
        "-lgpr",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "grpc++_unsecure",
    linkopts = [
        "-lgrpc++_unsecure",
        "-lgpr",
    ],
    visibility = ["//visibility:public"],
)

genrule(
    name = "ln_grpc_cpp_plugin",
    outs = ["grpc_cpp_plugin.bin"],
    cmd = "ln -s $$(which grpc_cpp_plugin) $@",
)

sh_binary(
    name = "grpc_cpp_plugin",
    srcs = ["grpc_cpp_plugin.bin"],
    visibility = ["//visibility:public"],
)

genrule(
    name = "ln_grpc_python_plugin",
    outs = ["grpc_python_plugin.bin"],
    cmd = "ln -s $$(which grpc_python_plugin) $@",
)

sh_binary(
    name = "grpc_python_plugin",
    srcs = ["grpc_python_plugin.bin"],
    visibility = ["//visibility:public"],
)
