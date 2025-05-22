# Copyright 2025 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrappers around pywrap rules for JAX."""

# NO_VISIBILITY_DECLARATION=.bzl file is intentionally exported to, e.g., JAX.

load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load(
    "//third_party/py/rules_pywrap:pywrap.impl.bzl",
    "pybind_extension",
    _pywrap_binaries = "pywrap_binaries",
    _pywrap_library = "pywrap_library",
)

pywrap_library = _pywrap_library
pywrap_binaries = _pywrap_binaries

def nanobind_pywrap_extension(
        name,
        srcs = [],
        deps = [],
        pytype_srcs = [],
        pytype_deps = [],  # @unused
        copts = [],
        linkopts = [],
        visibility = None):
    # buildifier: disable=function-docstring-args
    "Python extension rule using nanobind and the pywrap rules."
    module_name = name
    lib_name = name + "_pywrap_library"
    src_cc_name = name + "_pywrap_stub.c"

    # We put the entire contents of the extension in a single cc_library, which will become part of
    # the common pywrap library. All the contents of all extensions will end up in the common
    # library.
    native.cc_library(
        name = lib_name,
        srcs = srcs,
        copts = copts,
        deps = deps,
        local_defines = [
            "PyInit_{}=Wrapped_PyInit_{}".format(module_name, module_name),
        ],
        visibility = ["//visibility:private"],
    )

    # We build a small stub library as the extension that forwards to the PyInit_... symbol from the
    # common pywrap library.
    expand_template(
        name = name + "_pywrap_stub",
        testonly = True,
        out = src_cc_name,
        substitutions = {
            "@MODULE_NAME@": module_name,
        },
        template = "//xla/python:pyinit_stub.c",
        visibility = ["//visibility:private"],
    )

    # Despite its name "pybind_extension" has nothing to do with pybind. It is the Python extension
    # rule from the pywrap rules.
    pybind_extension(
        name = name,
        srcs = [src_cc_name],
        deps = [":" + lib_name],
        data = pytype_srcs,
        linkopts = linkopts,
        visibility = visibility,
        default_deps = [],
        common_lib_packages = [
            "jaxlib",
        ],
    )
