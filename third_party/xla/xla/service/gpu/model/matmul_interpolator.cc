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

#include "xla/service/gpu/model/matmul_interpolator.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/interpolator.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {
namespace {

struct InterpolationSpecification {
  int b;
  int m;
  int k;
  int n;
};

InterpolationSpecification ExtractDotSpec(const DotDimensionNumbers& dot_dims,
                                          const Shape& lhs, const Shape& rhs) {
  int b = 1, m = 1, n = 1, k = 1;
  for (int dim : dot_dims.lhs_batch_dimensions()) {
    b *= ShapeUtil::GetDimension(lhs, dim);
  }
  k *= ShapeUtil::ByteSizeOfPrimitiveType(lhs.element_type());
  for (int dim : dot_dims.lhs_contracting_dimensions()) {
    k *= ShapeUtil::GetDimension(lhs, dim);
  }
  m *= ShapeUtil::ByteSizeOfPrimitiveType(lhs.element_type());
  for (int dim :
       GetNonContractingDims(lhs.rank(), dot_dims.lhs_contracting_dimensions(),
                             dot_dims.lhs_batch_dimensions())) {
    m *= ShapeUtil::GetDimension(lhs, dim);
  }
  n *= ShapeUtil::ByteSizeOfPrimitiveType(rhs.element_type());
  for (int dim :
       GetNonContractingDims(rhs.rank(), dot_dims.rhs_contracting_dimensions(),
                             dot_dims.rhs_batch_dimensions())) {
    n *= ShapeUtil::GetDimension(rhs, dim);
  }
  return InterpolationSpecification{
      /*b=*/b,
      /*m=*/m,
      /*k=*/k,
      /*n=*/n,
  };
}

absl::StatusOr<InterpolationSpecification> Spec(
    const HloInstructionProfile& profile,
    const se::DeviceDescription& device_info) {
  if (profile.operands_size() != 2) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Expected exactly two operands for dot: ", profile.DebugString()));
  }
  if (profile.instruction().opcode() != HloOpcodeString(HloOpcode::kDot)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Expected dot, got: ", profile.instruction().DebugString()));
  }

  Shape lhs_shape = Shape(profile.operands(0).shape());
  Shape rhs_shape = Shape(profile.operands(1).shape());
  DotDimensionNumbers dot_dims = profile.instruction().dot_dimension_numbers();
  return ExtractDotSpec(dot_dims, lhs_shape, rhs_shape);
}

InterpolationSpecification Spec(const HloDotInstruction& dot) {
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  DotDimensionNumbers dot_dims = dot.dot_dimension_numbers();
  return ExtractDotSpec(dot_dims, lhs_shape, rhs_shape);
}

}  // namespace

/*static*/ absl::StatusOr<std::unique_ptr<MatmulInterpolator>>
MatmulInterpolator::Create(const HloInstructionProfileList& profiles,
                           const se::DeviceDescription& device_info) {
  auto interpolator = std::make_unique<EuclideanNNInterpolator<int64_t, 4>>();
  for (auto& profile : profiles.entries()) {
    TF_ASSIGN_OR_RETURN(InterpolationSpecification spec,
                        Spec(profile, device_info));
    std::array<int64_t, 4> point = {
        spec.b,
        spec.m,
        spec.k,
        spec.n,
    };
    int64_t fmas = 2ll * spec.b * spec.m * spec.n * spec.k;
    int64_t runtime_ns = profile.clock_cycles() / device_info.clock_rate_ghz();
    interpolator->Add(point, fmas * 1e9 / runtime_ns);
  }

  return std::unique_ptr<MatmulInterpolator>(
      new MatmulInterpolator(std::move(interpolator)));
}

std::optional<absl::Duration> MatmulInterpolator::EstimatedRuntime(
    const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kDot) {
    VLOG(1) << "Opcodes different than 'kDot' unsupported: "
            << instr.ToString();
    return std::nullopt;
  }
  auto* dot = Cast<HloDotInstruction>(&instr);
  InterpolationSpecification spec = Spec(*dot);
  std::array<int64_t, 4> point = {
      spec.b,
      spec.m,
      spec.k,
      spec.n,
  };
  int64_t flops = 2ll * spec.b * spec.m * spec.k * spec.n;
  return absl::Seconds(1.0 * flops / interpolator_->Eval(point));
}

}  // namespace xla::gpu
