# -*- Python -*-
# ruff: noqa: F821

import os

import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner

# (config is an instance of TestingConfig created when discovering tests)
# name: The name of this test suite
config.name = 'TRITON'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir', '.ll']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.triton_obj_root, 'test')
config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(("%shlibdir", config.llvm_shlib_dir))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

# llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.triton_obj_root, 'test')
config.triton_tools_dir = os.path.join(config.triton_obj_root, 'bin')
config.filecheck_dir = os.path.join(config.triton_obj_root, 'bin', 'FileCheck')

# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment["FILECHECK_OPTS"] = "--enable-var-scope"

tool_dirs = [config.triton_tools_dir, config.llvm_tools_dir, config.filecheck_dir]

# Tweak the PATH to include the tools dir.
for d in tool_dirs:
    llvm_config.with_environment('PATH', d, append_path=True)
tools = [
    'triton-opt',
    'triton-llvm-opt',
    ToolSubst('%PYTHON', config.python_executable, unresolved='ignore'),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# TODO: what's this?
llvm_config.with_environment('PYTHONPATH', [
    os.path.join(config.mlir_binary_dir, 'python_packages', 'triton'),
], append_path=True)
