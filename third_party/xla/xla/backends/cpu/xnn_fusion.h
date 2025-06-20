/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_CPU_XNN_FUSION_H_
#define XLA_BACKENDS_CPU_XNN_FUSION_H_

#include "xnnpack.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

inline constexpr absl::string_view kXnnFusionKind = "__xnn_fusion";

// Returns true if XNNPACK should use thread pool to execute given HLO
// instruction or computation. We rely on simple heuristics to determine if
// thread pool is beneficial.
bool XnnShouldUseThreadPool(const HloInstruction* hlo);
bool XnnShouldUseThreadPool(const HloComputation* computation);

// Returns true if the dot operation is supported by XNNPACK. Returns an error
// if the dot operation shape is invalid.
absl::StatusOr<bool> IsDotSupportedByXnn(
    const DotDimensionNumbers& dot_dimensions, const Shape& lhs_shape,
    const Shape& rhs_shape, const Shape& out_shape,
    const TargetMachineFeatures* cpu_features = nullptr);

absl::StatusOr<xnn_datatype> XnnDatatype(const PrimitiveType& type);
absl::StatusOr<xnn_unary_operator> XnnUnaryOperator(const HloOpcode& opcode);
absl::StatusOr<xnn_binary_operator> XnnBinaryOperator(const HloOpcode& opcode);

// Returns true if the shape either doesn't have a layout or the layout is
// descending. Shapes without layout are accepted to make HLO tests less
// verbose.
bool IsLayoutSupportedByXnn(const Shape& shape);

// Returns true if the constant is supported by XNNPACK.
bool IsConstantSupportedByXnn(const HloInstruction* hlo);

// Returns true if the nonconstant elementwise op is supported by XNNPACK.
bool IsElementwiseOpSupportedByXnn(const HloInstruction* hlo);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_XNN_FUSION_H_
