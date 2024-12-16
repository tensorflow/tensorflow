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
# ==============================================================================
"""Base classes for running kernels."""

import numpy as np

from xla.codegen.testlib import kernel_runner_extention
from xla.python import xla_extension

# Classes first
# go/keep-sorted start
ComparisonDirection = kernel_runner_extention.ComparisonDirection
DummyAddKernelRunner = kernel_runner_extention.DummyAddKernelRunner
HloInstruction = kernel_runner_extention.HloInstruction
HloOpcode = kernel_runner_extention.HloOpcode
KernelEmmitter = kernel_runner_extention.KernelEmitter
KernelRunner = kernel_runner_extention.KernelRunner
KernelSpec = kernel_runner_extention.KernelSpec
# go/keep-sorted end

# Functions
# go/keep-sorted start
opcode_arity = kernel_runner_extention.opcode_arity
# go/keep-sorted end


def create_literal_from_np(array: np.ndarray) -> xla_extension.Literal:
  shape = xla_extension.Shape.array_shape(array.dtype, array.shape)
  literal = xla_extension.Literal(shape)
  np.copyto(np.asarray(literal), array)
  return literal
