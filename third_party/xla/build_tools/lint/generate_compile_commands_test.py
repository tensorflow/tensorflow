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
from absl.testing import absltest

from xla.build_tools.lint import generate_compile_commands

CompileCommand = generate_compile_commands.CompileCommand


class CompileCommandsTest(absltest.TestCase):

  def test_command_from_args_list(self):
    arguments = [
        "/usr/bin/gcc",
        "-DTEST_DEFINE",
        "-fstack-protector",
        "-c",
        "xla/compiler.cc",
        "-o",
        "bazel-out/k8-opt/bin/xla/_objs/compiler/compiler.pic.o",
    ]

    command = CompileCommand.from_args_list(arguments)

    self.assertEqual(command.file, "xla/compiler.cc")
    self.assertEqual(command.arguments, arguments)

  def test_command_from_args_list_with_disallowed_option(self):
    arguments = [
        "/usr/bin/gcc",
        "-DTEST_DEFINE",
        "-fno-canonical-system-headers",
        "-c",
        "xla/compiler.cc",
        "-o",
        "bazel-out/k8-opt/bin/xla/_objs/compiler/compiler.pic.o",
    ]

    command = CompileCommand.from_args_list(arguments)

    self.assertEqual(command.file, "xla/compiler.cc")
    self.assertEqual(command.arguments, arguments[0:2] + arguments[3:])
