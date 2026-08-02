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

"""Private defs for this directory."""

# Intended only for use within this directory.
# Generated python wrappers are "private" visibility, users should depend on the
# full python code that incorporates the wrappers.  The generated targets have
# a _gen suffix, so that the full python version can use the bare name.
# We also hard code the hidden_file here to reduce duplication.
#
# We should consider moving the "out" default pattern into here, many other
# consumers of the tf_gen_op_wrapper_py rule would be simplified if we don't
# hard code the ops/ directory.

load("@xla//third_party/rules_python/python:defs.bzl", "py_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

# This is a private function only intended to be used in this directory, no need to
# document all its args for public consumption.
# buildifier: disable=function-docstring
def tf_gen_op_wrapper_private_py(
        name,
        out = None,
        deps = [],
        require_shape_functions = False,
        visibility = [],
        py_lib_rule = py_library):
    if not name.endswith("_gen"):
        fail("name must end in _gen")
    new_name = name[:-4]
    if not out:
        out = "gen_" + new_name + ".py"
    tf_gen_op_wrapper_py(
        name = new_name,  # Strip off _gen
        out = out,
        visibility = visibility or ["//visibility:private"],
        deps = deps,
        require_shape_functions = require_shape_functions,
        generated_target_name = name,
        extra_py_deps = [
            "//tensorflow/python:pywrap_tfe",
            "//tensorflow/python/util:dispatch",
            "//tensorflow/python/util:deprecation",
            "//tensorflow/python/util:tf_export",
        ],
        api_def_srcs = [
            "//tensorflow/core/api_def:base_api_def",
            "//tensorflow/core/api_def:python_api_def",
        ],
        py_lib_rule = py_lib_rule,
    )

def tf_gen_op_strict_wrapper_private_py(**kwargs):
    tf_gen_op_wrapper_private_py(py_lib_rule = py_library, **kwargs)
