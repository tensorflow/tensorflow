# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Build macros for checking ABI compatibility."""

load("@flatbuffers//:build_defs.bzl", "flatc_path")
load("@rules_shell//shell:sh_test.bzl", "sh_test")

def flatbuffer_schema_compat_test(name, ref_schema, schema):
    """Generates a test for schema binary compatibility.

    Generates a test that the specified schema file is binary backwards
    compatible with a reference schema (e.g. a previous version of the
    schema).

    Note: currently this build macro requires that the schema be a single
    fully self-contained .fbs file; it does not yet support includes.
    """

    native.genrule(
        name = name + "_gen",
        srcs = [ref_schema, schema],
        outs = [name + "_test.sh"],
        tools = [flatc_path],
        cmd = ("echo $(rootpath {}) --conform $(rootpath {}) $(rootpath {}) > $@"
            .format(flatc_path, ref_schema, schema)),
    )
    sh_test(
        name = name,
        srcs = [name + "_test.sh"],
        data = [flatc_path, ref_schema, schema],
    )
