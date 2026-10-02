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

"""A simple portable implementation of build_test."""

load("@rules_shell//shell:sh_test.bzl", "sh_test")

def build_test(name, targets, visibility = None):
    """Generates a test that just verifies that the specified targets can be built."""

    # Generate an sh_test rule that lists the specified targets as data,
    # (thus forcing those targets to be built before the test can be run)
    # and that runs a script which always succeeds.
    sh_test(
        name = name,
        srcs = [name + ".sh"],
        data = targets,
        visibility = visibility,
    )

    # Generate the script which always succeeds.  We just generate an empty script.
    native.genrule(
        name = name + "_gen_sh",
        outs = [name + ".sh"],
        cmd = "> $@",
        visibility = ["//visibility:private"],
    )
