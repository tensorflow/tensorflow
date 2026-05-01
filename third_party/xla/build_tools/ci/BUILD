# Copyright 2024 The OpenXLA Authors.
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
# ============================================================================

load("@bazel_skylib//rules:diff_test.bzl", "diff_test")
load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
load("//xla:pytype.bzl", "pytype_strict_binary")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    licenses = ["notice"],
)

pytype_strict_binary(
    name = "build",
    srcs = ["build.py"],
)

genrule(
    name = "generated_build_commands",
    outs = ["generated_commands.txt"],
    cmd = "$(location //build_tools/ci:build) --dump_commands > $(OUTS)",
    tags = ["not_run:arm"],
    tools = [":build"],
)

diff_test(
    name = "build_command_golden_test",
    failure_message = """Regenerate with `PYTHONDONTWRITEBYTECODE=1 python3 build.py --dump_commands > golden_commands.txt`.""",
    file1 = "golden_commands.txt",
    file2 = ":generated_build_commands",
    tags = ["not_run:arm"],
)

sh_binary(
    name = "parallel_gpu_execute",
    srcs = ["parallel_gpu_execute.sh"],
)

# TODO: When upstream rules expose clang-tidy use that instead of explicitly creating one.
_LLVM_REPO_NAME = "llvm18_linux_x86_64"

expand_template(
    name = "generate_wrapper_script",
    out = "clang_tidy_wrapper.sh",
    substitutions = {
        "%LLVM_REPO_NAME%": _LLVM_REPO_NAME,
    },
    template = "clang_tidy_wrapper.sh.tpl",
)

sh_binary(
    name = "clang_tidy_bin",
    srcs = [":generate_wrapper_script"],
    data = [
        # This not for g3. Only for running clang-tidy on CI/locally for OSS workflows.
        "@{}//:all".format(_LLVM_REPO_NAME),  # buildifier: disable=platform-specific-binaries
    ],
    tags = [
        "manual",  # This binary only runs via explicit targeting.
        "notap",  # Not for g3.
    ],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "run_clang_tidy",
    srcs = ["run_clang_tidy.sh"],
)
