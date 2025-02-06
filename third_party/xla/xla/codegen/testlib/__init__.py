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
"""Public API for codegen testlib."""

from xla.codegen.testlib import _extension

# Classes
# go/keep-sorted start
BufferAssignment = _extension.BufferAssignment
ComparisonDirection = _extension.ComparisonDirection
DotDimensionNumbers = _extension.DotDimensionNumbers
HloInstruction = _extension.HloInstruction
HloModule = _extension.HloModule
HloModuleConfig = _extension.HloModuleConfig
HloOpcode = _extension.HloOpcode
KernelDefinition = _extension.KernelDefinition
KernelEmmitter = _extension.KernelEmitter
KernelRunner = _extension.KernelRunner
KernelSpec = _extension.KernelSpec
# go/keep-sorted end
