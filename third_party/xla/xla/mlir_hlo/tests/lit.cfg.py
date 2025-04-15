"""Lit configuration to drive test in this repo."""
# Copyright 2020 The OpenXLA Authors.
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

# -*- Python -*-
# pylint: disable=undefined-variable

import os
import sys

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
import lit.util

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'MLIR_HLO_OPT'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

# Adjusted the PATH to correctly detect the tools on Windows
if sys.platform == 'win32':
  llvm_config.config.llvm_tools_dir = r'..\llvm-project\llvm'
  llvm_config.config.mlir_binary_dir = r'..\llvm-project\mlir'

llvm_config.use_default_substitutions()

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [
    config.mlir_binary_dir,
    config.mlir_hlo_tools_dir,
]
tools = [
    'mlir-hlo-opt',
    'mlir-runner',
    ToolSubst('%mlir_lib_dir', config.mlir_lib_dir, unresolved='ignore'),
]
llvm_config.add_tool_substitutions(tools, tool_dirs)
