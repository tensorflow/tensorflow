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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import lit.llvm

# Lint for undefined variables is disabled as config is not defined inside this
# file, instead config is injected by lit.py. The structure is common for lit
# tests and intended to only persist temporarily (b/136126535).
# pylint: disable=undefined-variable
config.llvm_tools_dir = os.path.join(os.environ['TEST_SRCDIR'], 'llvm-project',
                                     'llvm')
config.mlir_obj_root = os.path.join(os.environ['TEST_SRCDIR'])
config.mlir_tools_dir = os.path.join(os.environ['TEST_SRCDIR'], 'lllvm-project',
                                     'mlir')
# TODO(jpienaar): Replace with suffices in build rule.
config.suffixes = ['.td', '.mlir', '.pbtxt']

mlir_tf_tools_dirs = [
    'tensorflow/compiler/mlir',
    'tensorflow/compiler/mlir/lite',
    'tensorflow/compiler/mlir/tensorflow',
    'tensorflow/compiler/mlir/xla',
]
config.mlir_tf_tools_dirs = [
    os.path.join(os.environ['TEST_SRCDIR'], os.environ['TEST_WORKSPACE'], s)
    for s in mlir_tf_tools_dirs
]
test_dir = os.environ['TEST_TARGET']
test_dir = test_dir.strip('/').rsplit(':', 1)[0]
config.mlir_test_dir = os.path.join(os.environ['TEST_SRCDIR'],
                                    os.environ['TEST_WORKSPACE'], test_dir)
lit.llvm.initialize(lit_config, config)

# Let the main config do the real work.
lit_config.load_config(
    config,
    os.path.join(
        os.path.join(os.environ['TEST_SRCDIR'], os.environ['TEST_WORKSPACE'],
                     'tensorflow/compiler/mlir/runlit.cfg.py')))
# pylint: enable=undefined-variable
