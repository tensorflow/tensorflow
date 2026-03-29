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

#include "xla/backends/cpu/onednn_support.h"

#include "absl/base/no_destructor.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "oneapi/dnnl/dnnl.hpp"  // NOLINT: for DNNL_MAX_NDIMS
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/dot_dims.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"

namespace xla::cpu {

bool IsOneDnnSupportedDType(PrimitiveType dtype,
                            const TargetMachineFeatures* cpu_features) {
  if (dtype == F32 || IsSupportedType(dtype)) {
    return true;
  }
  // Check for data type support if target machine features are provided.
  // Unit tests may provide target machine features to simulate different CPU
  // capabilities.
  return (cpu_features != nullptr &&
          ((dtype == BF16 && cpu_features->has_avx512bf16()) ||
           (dtype == F16 && cpu_features->has_avx512fp16())));
}

bool IsOneDnnSupportedLayout(const Shape& shape) {
  return !shape.has_layout() || LayoutUtil::HasDescendingLayout(shape.layout());
}

bool IsOneDnnSupportedTypeAndLayout(const HloInstruction* hlo,
                                    const TargetMachineFeatures* cpu_features) {
  auto is_supported = [cpu_features](const HloInstruction* hlo) {
    return IsOneDnnSupportedDType(hlo->shape().element_type(), cpu_features) &&
           IsOneDnnSupportedLayout(hlo->shape());
  };

  if (!is_supported(hlo)) {
    return false;
  }
  return (std::all_of(hlo->operands().begin(), hlo->operands().end(),
                      is_supported));
}

absl::StatusOr<bool> IsDotSupportedByOneDnn(
    const DotDimensionNumbers& dot_dimensions, const Shape& lhs_shape,
    const Shape& rhs_shape, const Shape& out_shape,
    const TargetMachineFeatures* cpu_features) {
  if (lhs_shape.element_type() != rhs_shape.element_type() ||
      lhs_shape.element_type() != out_shape.element_type()) {
    return false;
  }
  if (!IsOneDnnSupportedDType(out_shape.element_type(), cpu_features)) {
    return false;
  }

  if (ShapeUtil::IsZeroElementArray(lhs_shape) ||
      ShapeUtil::IsZeroElementArray(rhs_shape) ||
      ShapeUtil::IsZeroElementArray(out_shape)) {
    return false;
  }

  // NOLINTNEXTLINE: Use dnnl.hpp for DNNL_MAX_NDIMS for now.
  if (lhs_shape.dimensions().size() > DNNL_MAX_NDIMS ||
      rhs_shape.dimensions().size() > DNNL_MAX_NDIMS ||
      lhs_shape.dimensions().size() != rhs_shape.dimensions().size()) {
    return false;
  }

  auto dot_shape_result =
      GetDotShape(dot_dimensions, lhs_shape, rhs_shape, out_shape);
  if (!dot_shape_result.ok()) {
    VLOG(2) << "GetDotShape Error: " << dot_shape_result.status();
    return false;
  }
  DotShape dot_shape = dot_shape_result.value();

  auto dot_canonical_result = GetDotCanonicalDims(dot_dimensions, dot_shape);
  if (!dot_canonical_result.ok()) {
    VLOG(2) << "GetDotCanonicalDims Error: " << dot_canonical_result.status();
    return false;
  }
  DotCanonicalDims dot_canonical_dims = dot_canonical_result.value();

  // Restrict support to row-major layouts.
  return !dot_canonical_dims.lhs_column_major &&
         !dot_canonical_dims.rhs_column_major;
}

const absl::flat_hash_map<HloOpcode, op::kind>& GetOneDnnUnaryOpMap() {
  static absl::NoDestructor<absl::flat_hash_map<HloOpcode, op::kind>>
      unary_op_map({
          {HloOpcode::kAbs, op::kind::Abs},
          {HloOpcode::kExp, op::kind::Exp},
          {HloOpcode::kLog, op::kind::Log},
          {HloOpcode::kSqrt, op::kind::Sqrt},
          {HloOpcode::kTanh, op::kind::Tanh},
      });
  return *unary_op_map;
}

absl::StatusOr<op::kind> OneDnnUnaryOperator(const HloOpcode& opcode) {
  const auto& unary_op_map = GetOneDnnUnaryOpMap();
  auto result = unary_op_map.find(opcode);
  if (result == unary_op_map.end()) {
    return InvalidArgument("Unsupported OneDNN unary operator: %s",
                           HloOpcodeString(opcode));
  }
  return result->second;
}

std::vector<absl::string_view> GetOneDnnSupportedUnaryOpsStrings() {
  auto& unary_op_map = GetOneDnnUnaryOpMap();
  std::vector<absl::string_view> op_names;
  op_names.reserve(unary_op_map.size());
  for (auto& pair : unary_op_map) {
    op_names.push_back(HloOpcodeString(pair.first));
  }
  return op_names;
}

const absl::flat_hash_map<HloOpcode, op::kind>& GetOneDnnBinaryOpMap() {
  static absl::NoDestructor<absl::flat_hash_map<HloOpcode, op::kind>>
      binary_op_map({
          {HloOpcode::kAdd, op::kind::Add},
          {HloOpcode::kDivide, op::kind::Divide},
          {HloOpcode::kDot, op::kind::MatMul},
          {HloOpcode::kMaximum, op::kind::Maximum},
          {HloOpcode::kMinimum, op::kind::Minimum},
          {HloOpcode::kMultiply, op::kind::Multiply},
          {HloOpcode::kSubtract, op::kind::Subtract},
      });
  return *binary_op_map;
}

absl::StatusOr<op::kind> OneDnnBinaryOperator(const HloOpcode& opcode) {
  const auto& binary_op_map = GetOneDnnBinaryOpMap();
  auto result = binary_op_map.find(opcode);
  if (result == binary_op_map.end()) {
    return InvalidArgument("Unsupported OneDNN binary operator: %s",
                           HloOpcodeString(opcode));
  }
  return result->second;
}

std::vector<absl::string_view> GetOneDnnSupportedBinaryOpsStrings() {
  auto& binary_op_map = GetOneDnnBinaryOpMap();
  std::vector<absl::string_view> op_names;
  op_names.reserve(binary_op_map.size());
  for (auto& pair : binary_op_map) {
    op_names.push_back(HloOpcodeString(pair.first));
  }
  return op_names;
}

bool IsOpSupportedByOneDnn(const HloInstruction* hlo,
                           const TargetMachineFeatures* cpu_features) {
  if (!OneDnnBinaryOperator(hlo->opcode()).ok() &&
      !OneDnnUnaryOperator(hlo->opcode()).ok()) {
    return false;
  }
  if (hlo->opcode() == HloOpcode::kDot) {
    return IsDotSupportedByOneDnn(
               hlo->dot_dimension_numbers(), hlo->operand(0)->shape(),
               hlo->operand(1)->shape(), hlo->shape(), cpu_features)
        .value_or(false);
  }
  if (hlo->opcode() == HloOpcode::kBitcast) {
    return IsBitcastOpSupportedByOneDnn(hlo, cpu_features);
  }

  return IsOneDnnSupportedTypeAndLayout(hlo, cpu_features);
}

bool IsConstantSupportedByOneDnn(const HloInstruction* hlo,
                                 const TargetMachineFeatures* cpu_features) {
  CHECK(hlo->IsConstant());
  return IsOneDnnSupportedDType(hlo->shape().element_type(), cpu_features) &&
         IsOneDnnSupportedLayout(hlo->shape());
}

bool IsBitcastOpSupportedByOneDnn(const HloInstruction* hlo,
                                  const TargetMachineFeatures* cpu_features) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kBitcast);
  if (!IsOneDnnSupportedTypeAndLayout(hlo, cpu_features)) {
    return false;
  }
  const HloInstruction* input = hlo->operand(0);
  return hlo->shape().element_type() == input->shape().element_type();
}

bool IsElementwiseOpSupportedByOneDnn(
    const HloInstruction* hlo, const TargetMachineFeatures* cpu_features) {
  CHECK(hlo->IsElementwise());
  if (hlo->IsConstant()) {
    return IsConstantSupportedByOneDnn(hlo, cpu_features);
  }
  if (hlo->opcode() == HloOpcode::kBitcast) {
    return IsBitcastOpSupportedByOneDnn(hlo, cpu_features);
  }
  if (!OneDnnBinaryOperator(hlo->opcode()).ok() &&
      !OneDnnUnaryOperator(hlo->opcode()).ok()) {
    return false;
  }
  return IsOneDnnSupportedTypeAndLayout(hlo, cpu_features);
}

}  // namespace xla::cpu
