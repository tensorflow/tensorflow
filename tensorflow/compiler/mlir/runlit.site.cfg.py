# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Lit runner site configuration."""

import os
import platform
import lit.llvm

# Handle the test srcdir for platforms. On windows, things are weird with bazel.
if platform.system() == 'Windows':
  srcdir = os.environ['TEST_SRCDIR']
  real_test_srcdir = srcdir[:srcdir.find('tensorflow/compiler/mlir')]
  external_srcdir = os.path.join(real_test_srcdir, 'external')
else:
  real_test_srcdir = os.environ['TEST_SRCDIR']
  external_srcdir = real_test_srcdir

# Lint for undefined variables is disabled as config is not defined inside this
# file, instead config is injected by lit.py. The structure is common for lit
# tests and intended to only persist temporarily (b/136126535).
# pylint: disable=undefined-variable
config.llvm_tools_dir = os.path.join(external_srcdir, 'llvm-project', 'llvm')
config.mlir_obj_root = os.path.join(real_test_srcdir)
config.mlir_tools_dir = os.path.join(external_srcdir, 'llvm-project', 'mlir')
# TODO(jpienaar): Replace with suffices in build rule.
config.suffixes = ['.td', '.mlir', '.pbtxt']

mlir_tf_tools_dirs = [
    'tensorflow/core/ir/importexport/',
    'tensorflow/core/ir/tests/',
    'tensorflow/core/transforms/',
    'tensorflow/compiler/mlir',
    'tensorflow/compiler/xla/mlir_hlo',
    'tensorflow/compiler/xla/mlir_hlo/tosa',
    'tensorflow/compiler/mlir/lite',
    'tensorflow/compiler/mlir/lite/experimental/tac',
    'tensorflow/compiler/mlir/quantization/tensorflow',
    'tensorflow/compiler/mlir/tensorflow',
    'tensorflow/compiler/mlir/tfrt',
    'tensorflow/compiler/mlir/xla',
    'tensorflow/compiler/mlir/tools/kernel_gen',
    'tensorflow/compiler/aot',
    'tensorflow/compiler/xla/service/mlir_gpu',
    'tensorflow/compiler/xla/service/gpu/tests',
    'tensorflow/compiler/xla/mlir/tools/runtime',
    'tensorflow/compiler/mlir/lite/stablehlo',
]
config.mlir_tf_tools_dirs = [
    os.path.join(real_test_srcdir, os.environ['TEST_WORKSPACE'], s)
    for s in mlir_tf_tools_dirs
]
test_dir = os.environ['TEST_TARGET']
test_dir = test_dir.strip('/').rsplit(':', 1)[0]
config.mlir_test_dir = os.path.join(real_test_srcdir,
                                    os.environ['TEST_WORKSPACE'], test_dir)

if platform.system() == 'Windows':
  # Configure this to work with msys2, TF's preferred windows bash.
  config.lit_tools_dir = '/usr/bin'

lit.llvm.initialize(lit_config, config)

# Let the main config do the real work.
lit_config.load_config(
    config,
    os.path.join(
        os.path.join(real_test_srcdir, os.environ['TEST_WORKSPACE'],
                     'tensorflow/compiler/mlir/runlit.cfg.py')))
# pylint: enable=undefined-variable
