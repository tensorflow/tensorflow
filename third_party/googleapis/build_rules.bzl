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
Utilities for building grpc and proto libraries from googleapis.
"""

load("@rules_cc//cc:defs.bzl", native_cc_proto_library = "cc_proto_library")
load("@com_github_grpc_grpc//bazel:generate_cc.bzl", "generate_cc")

def _tf_cc_headers(ctx):
    if len(ctx.attr.deps) != 1:
        fail("deps must have exactly 1 photo_library")
    return [
        CcInfo(
            compilation_context = ctx.attr.deps[0][CcInfo].compilation_context,
        ),
        DefaultInfo(
            files = ctx.attr.deps[0][CcInfo].compilation_context.headers,
        ),
    ]

tf_cc_headers = rule(
    implementation = _tf_cc_headers,
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
    },
)

def cc_proto_library(name, deps):
    """Generates a cc library and a header only cc library from a proto library

    Args:
      name: the name of the cc_library
      deps: a list that contains exactly one proto_library
    """
    native_cc_proto_library(
        name = name,
        deps = deps,
        visibility = ["//visibility:public"],
    )
    tf_cc_headers(
        name = name + "_headers_only",
        deps = [":" + name],
        visibility = ["//visibility:public"],
    )

def cc_grpc_library(name, srcs, deps, **kwargs):
    """Generates a cc library with grpc implementation and cc proto headers

    Args:
      name: the name of the cc_grpc_library to be created
      srcs: the proto_libraries used to generate the cc_grpc_library
      deps: the dependencies used to link into this cc_grpc_library, defined by
        cc_proto_library
      **kwargs: other args not used, for compatibility only
    """
    if len(srcs) != 1:
        fail("srcs must have exactly 1 photo_library", "srcs")
    codegen_grpc_target = "_" + name + "_grpc_codegen"
    generate_cc(
        name = codegen_grpc_target,
        srcs = srcs,
        plugin = "@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin",
        well_known_protos = True,
        generate_mocks = True,
    )

    grpc_proto_dep = "@com_github_grpc_grpc//:grpc++_codegen_proto"
    native.cc_library(
        name = name,
        srcs = [":" + codegen_grpc_target],
        hdrs = [":" + codegen_grpc_target],
        deps = [dep + "_headers_only" for dep in deps] + [grpc_proto_dep],
        visibility = ["//visibility:public"],
    )
