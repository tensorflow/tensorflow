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
"""Public API for cpu codegen testlib."""

from xla.backends.cpu.testlib import _extension

# go/keep-sorted start
ConcatenateKernelEmitter = _extension.ConcatenateKernelEmitter
DotKernelEmitter = _extension.DotKernelEmitter
ElementalKernelEmitter = _extension.ElementalKernelEmitter
HloCompiler = _extension.HloCompiler
JitCompiler = _extension.JitCompiler
KernelRunner = _extension.KernelRunner
LlvmIrKernelEmitter = _extension.LlvmIrKernelEmitter
TargetMachineFeatures = _extension.TargetMachineFeatures
# go/keep-sorted end
