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

#ifndef XLA_BACKENDS_CPU_YNN_SUPPORT_H_
#define XLA_BACKENDS_CPU_YNN_SUPPORT_H_

#include "ynnpack/include/ynnpack.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla::cpu {

inline constexpr absl::string_view kYnnFusionKind = "__ynn_fusion";

// Returns the mappings from HLO opcodes to YNNPACK unary operators.
const absl::flat_hash_map<HloOpcode, ynn_unary_operator>& GetYnnUnaryOpMap();

// Returns the YNNPACK unary operator corresponding to the given HLO opcode.
// Returns `InvalidArgument` if the opcode is not supported.
absl::StatusOr<ynn_unary_operator> YnnUnaryOperator(const HloOpcode& opcode);

// Returns the mappings from HLO opcodes to YNNPACK binary operators.
const absl::flat_hash_map<HloOpcode, ynn_binary_operator>& GetYnnBinaryOpMap();

// Returns the YNNPACK binary operator corresponding to the given HLO opcode.
// Returns `InvalidArgument` if the opcode is not supported.
absl::StatusOr<ynn_binary_operator> YnnBinaryOperator(const HloOpcode& opcode);

// Returns true if the shape either doesn't have a layout or the layout is
// descending. Shapes without layout are accepted to make HLO tests less
// verbose.
bool IsLayoutSupportedByYnn(const Shape& shape);

// Returns true if the bitcast op is supported by YNNPACK.
bool IsBitcastOpSupportedByYnn(const HloInstruction* hlo);

// Returns true if the constant is supported by YNNPACK.
bool IsConstantSupportedByYnn(const HloInstruction* hlo);

// Returns true if the nonconstant elementwise op is supported by YNNPACK.
bool IsElementwiseOpSupportedByYnn(const HloInstruction* hlo);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_YNN_SUPPORT_H_
