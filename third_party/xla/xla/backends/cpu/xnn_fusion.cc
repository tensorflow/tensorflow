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

#include "xla/backends/cpu/xnn_fusion.h"

#include <algorithm>
#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// Thresholds for when to use thread pool for XNNPACK fusions for different
// HLOs. These numbers picked up randomly and need benchmarks to tune.
static constexpr int64_t kDotThreshold = 10 * 1000;
static constexpr int64_t kDefaultThreshold = 100 * 1000;

static int64_t MaxElementsCount(const Shape& shape) {
  int64_t ret = 0;
  ShapeUtil::ForEachSubshape(
      shape, [&](const Shape& shape, const ShapeIndex& index) {
        ret = std::max(ret, ShapeUtil::ElementsIn(shape));
      });
  return ret;
}

// We rely on a very simple heuristic to determine if thread pool is beneficial
// for XNNPACK fusions. We assume that if the HLO produces a large result (or
// has large operands), thread pool will be beneficial for running operation in
// parallel. For small operations, thread pool overheads are higher than the
// actual computation.
static int64_t MaxElementsCount(const HloInstruction* hlo,
                                bool include_operands = true) {
  int64_t ret = MaxElementsCount(hlo->shape());
  if (include_operands) {
    for (auto* operand : hlo->operands()) {
      ret = std::max(ret, MaxElementsCount(operand->shape()));
    }
  }
  return ret;
}

bool XnnShouldUseThreadPool(const HloInstruction* hlo) {
  switch (hlo->opcode()) {
    case HloOpcode::kDot:
      return MaxElementsCount(hlo) > kDotThreshold;
    default:
      return MaxElementsCount(hlo) > kDefaultThreshold;
  }
}

bool XnnShouldUseThreadPool(const HloComputation* computation) {
  return absl::c_any_of(
      computation->instructions(),
      [](const HloInstruction* hlo) { return XnnShouldUseThreadPool(hlo); });
}

absl::StatusOr<bool> IsXnnDotSupported(
    const DotDimensionNumbers& dot_dimensions, const Shape& lhs_shape,
    const Shape& rhs_shape, const Shape& out_shape,
    const TargetMachineFeatures* cpu_features) {
  // TODO(ezhulenev): Support other element types.
  auto check_dtype = [&](PrimitiveType in_dtype, PrimitiveType out_dtype) {
    return lhs_shape.element_type() == in_dtype &&
           rhs_shape.element_type() == in_dtype &&
           out_shape.element_type() == out_dtype;
  };

  // We assume that the feature is available if `cpu_features` is not provided.
  bool cpu_has_avx512bf16 =
      cpu_features == nullptr || cpu_features->has_avx512bf16();
  bool dtype_is_supported =
      check_dtype(F32, F32) || (check_dtype(BF16, F32) && cpu_has_avx512bf16);
  if (!dtype_is_supported) {
    return false;
  }

  TF_ASSIGN_OR_RETURN(DotShape dot_shape, GetDotShape(dot_dimensions, lhs_shape,
                                                      rhs_shape, out_shape));

  TF_ASSIGN_OR_RETURN(DotCanonicalDims dot_canonical_dims,
                      GetDotCanonicalDims(dot_dimensions, dot_shape));

  // TODO(b/385370486): XNNPACK does not tile by `K` and can be a lot slower
  // than the default Eigen implementation.
  if (dot_canonical_dims.k / dot_canonical_dims.m > 5 ||
      dot_canonical_dims.k / dot_canonical_dims.n > 5) {
    return false;
  }

  // XNNPACK does not support transposing LHS or col-major layouts.
  return dot_canonical_dims.lhs_canonical &&
         !dot_canonical_dims.lhs_column_major &&
         !dot_canonical_dims.rhs_column_major;
}

}  // namespace xla::cpu
