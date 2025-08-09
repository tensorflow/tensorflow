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
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/gpu/model/interpolator.h"
#include "xla/service/gpu/model/matmul_interpolator_data.h"
#include "xla/service/gpu/model/matmul_interpolator_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

struct InterpolationSpecification {
  int b;
  // Matmul dimensions are normalized to a number of bytes of output type.
  int m;
  int k;
  int n;
  // Data types.
  PrimitiveType lhs_type;
  PrimitiveType rhs_type;
  PrimitiveType out_type;
};

struct InterpolationSpecificationFlops {
  InterpolationSpecification dims;
  int64_t flops;
};

bool IsTritonGemm(const HloInstruction& instr) {
  if (instr.called_computations().size() != 1) {
    return false;
  }
  if (!IsTritonFusedComputation(*instr.called_computations()[0])) {
    return false;
  }
  auto fused_range = instr.fused_instructions();
  return absl::c_count_if(fused_range, HloPredicateIsOp<HloOpcode::kDot>) == 1;
}

InterpolationSpecification ExtractDotSpec(const DotDimensionNumbers& dot_dims,
                                          const Shape& lhs, const Shape& rhs,
                                          const Shape& out) {
  int b = 1, m = 1, n = 1, k = 1;
  for (int dim : dot_dims.lhs_batch_dimensions()) {
    b *= ShapeUtil::GetDimension(lhs, dim);
  }
  k *= ShapeUtil::ByteSizeOfPrimitiveType(out.element_type());
  for (int dim : dot_dims.lhs_contracting_dimensions()) {
    k *= ShapeUtil::GetDimension(lhs, dim);
  }
  m *= ShapeUtil::ByteSizeOfPrimitiveType(out.element_type());
  for (int dim : GetNonContractingDims(lhs.dimensions().size(),
                                       dot_dims.lhs_contracting_dimensions(),
                                       dot_dims.lhs_batch_dimensions())) {
    m *= ShapeUtil::GetDimension(lhs, dim);
  }
  n *= ShapeUtil::ByteSizeOfPrimitiveType(out.element_type());
  for (int dim : GetNonContractingDims(rhs.dimensions().size(),
                                       dot_dims.rhs_contracting_dimensions(),
                                       dot_dims.rhs_batch_dimensions())) {
    n *= ShapeUtil::GetDimension(rhs, dim);
  }
  return InterpolationSpecification{
      /*b=*/b,
      /*m=*/m,
      /*k=*/k,
      /*n=*/n,
      /*lhs_type=*/lhs.element_type(),
      /*rhs_type=*/rhs.element_type(),
      /*out_type=*/out.element_type(),
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

  TF_ASSIGN_OR_RETURN(Shape lhs_shape,
                      Shape::FromProto(profile.operands(0).shape()));
  TF_ASSIGN_OR_RETURN(Shape rhs_shape,
                      Shape::FromProto(profile.operands(1).shape()));
  DotDimensionNumbers dot_dims = profile.instruction().dot_dimension_numbers();
  TF_ASSIGN_OR_RETURN(Shape out_shape,
                      Shape::FromProto(profile.instruction().shape()));
  return ExtractDotSpec(dot_dims, lhs_shape, rhs_shape, out_shape);
}

InterpolationSpecification Spec(const HloDotInstruction& dot,
                                const Shape& out_shape) {
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  DotDimensionNumbers dot_dims = dot.dot_dimension_numbers();
  return ExtractDotSpec(dot_dims, lhs_shape, rhs_shape, out_shape);
}

InterpolationSpecification Spec(const HloCustomCallInstruction& dot,
                                const Shape& out_shape) {
  CHECK(IsCublasGemm(dot));
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  DotDimensionNumbers dot_dims = dot.backend_config<GpuBackendConfig>()
                                     ->gemm_backend_config()
                                     .dot_dimension_numbers();
  return ExtractDotSpec(dot_dims, lhs_shape, rhs_shape, out_shape);
}

InterpolationSpecification Spec(const HloFusionInstruction& dot_fusion,
                                const Shape& out_shape) {
  CHECK(IsTritonGemm(dot_fusion));

  auto fused = dot_fusion.fused_instructions();
  auto dot_it = absl::c_find_if(fused, HloPredicateIsOp<HloOpcode::kDot>);
  CHECK(dot_it != std::end(fused));

  const HloDotInstruction& dot = *Cast<HloDotInstruction>(*dot_it);
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  DotDimensionNumbers dot_dims = dot.dot_dimension_numbers();
  return ExtractDotSpec(dot_dims, lhs_shape, rhs_shape, out_shape);
}

absl::StatusOr<GemmPerfTableEntryValues> ReadDefaultProfile(
    const se::DeviceDescription& device_info) {
  GemmPerfTable table;
  if (!tsl::protobuf::TextFormat::ParseFromString(kDefaultMatmulPTable,
                                                  &table)) {
    return absl::FailedPreconditionError("Cannot parse a default perf table.");
  }
  std::string key = HloOpProfiles::GetProfileName(device_info);

  if (!table.entries().contains(key)) {
    return absl::NotFoundError(absl::StrCat("Cannot find key: ", key));
  }
  return table.entries().at(key);
}

std::optional<InterpolationSpecification> GetInterpolationSpec(
    const HloInstruction& instr) {
  std::optional<InterpolationSpecification> spec;
  if (instr.opcode() == HloOpcode::kDot) {
    auto* dot = Cast<HloDotInstruction>(&instr);
    spec = Spec(*dot, instr.shape());
  } else if (IsCublasGemm(instr)) {
    auto* dot = Cast<HloCustomCallInstruction>(&instr);
    spec = Spec(*dot, instr.shape().IsTuple()
                          ? ShapeUtil::GetTupleElementShape(instr.shape(), 0)
                          : instr.shape());
  } else if (IsTritonGemm(instr)) {
    auto* dot_fusion = Cast<HloFusionInstruction>(&instr);
    spec = Spec(*dot_fusion, instr.shape());
  }
  return spec;
}

absl::Duration PreciseFlopsRetrieval(
    const InterpolationSpecification& spec,
    const EuclideanWeightedAverageInterpolator<4>& interpolator) {
  // Denormalize back to normal dims.
  InterpolationSpecification denormalized_spec = spec;
  denormalized_spec.b = spec.b;
  denormalized_spec.m =
      spec.m / ShapeUtil::ByteSizeOfPrimitiveType(spec.out_type);
  denormalized_spec.n =
      spec.n / ShapeUtil::ByteSizeOfPrimitiveType(spec.out_type);
  denormalized_spec.k =
      spec.k / ShapeUtil::ByteSizeOfPrimitiveType(spec.out_type);
  std::array<int64_t, 4> point = {
      denormalized_spec.b,
      denormalized_spec.m,
      denormalized_spec.k,
      denormalized_spec.n,
  };
  int64_t flops = 2ll * denormalized_spec.b * denormalized_spec.m *
                  denormalized_spec.n * denormalized_spec.k;
  return absl::Seconds(1.0 * flops / interpolator.Eval(point));
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

/*static*/ absl::StatusOr<std::unique_ptr<MatmulInterpolator>>
MatmulInterpolator::Create(const se::DeviceDescription& device_info) {
  TF_ASSIGN_OR_RETURN(GemmPerfTableEntryValues table,
                      ReadDefaultProfile(device_info));
  absl::flat_hash_map<MatmulDTypeKey,
                      std::vector<InterpolationSpecificationFlops>>
      spec_map;
  for (const GemmPerfTableEntry& entry : table.entries()) {
    for (const auto& [dtype_key, flops] : entry.flops()) {
      MatmulDTypeKey key(dtype_key);
      InterpolationSpecificationFlops spec;
      spec.flops = flops;
      spec.dims.b = entry.b();
      spec.dims.m = entry.m();
      spec.dims.k = entry.k();
      spec.dims.n = entry.n();
      spec_map[key].push_back(spec);
    }
  }

  auto interpolators = std::make_unique<absl::flat_hash_map<
      MatmulDTypeKey,
      std::unique_ptr<EuclideanWeightedAverageInterpolator<4>>>>();
  std::array<int64_t, 4> max_context = {4, 4096, 4096, 4096};
  std::array<int64_t, 4> min_context = {1, 256, 256, 256};
  std::array<int64_t, 4> fixed_complement_off = {-1, -1, -1, -1};
  std::array<int64_t, 4> factor_of_two_complement_on = {1, 1, 1, 1};
  auto fallback_interpolator =
      std::make_unique<EuclideanWeightedAverageInterpolator<4>>(
          fixed_complement_off, factor_of_two_complement_on,
          /*max_context=*/std::array<int64_t, 4>{4, 8192, 8192, 8192},
          /*min_context=*/std::array<int64_t, 4>{1, 512, 512, 512});
  for (const auto& [dtype_key, specs] : spec_map) {
    (*interpolators)[dtype_key] =
        std::make_unique<EuclideanWeightedAverageInterpolator<4>>(
            fixed_complement_off, factor_of_two_complement_on, max_context,
            min_context);
    for (InterpolationSpecificationFlops spec : specs) {
      std::array<int64_t, 4> point = {
          spec.dims.b,
          spec.dims.m,
          spec.dims.k,
          spec.dims.n,
      };
      (*interpolators)[dtype_key]->Add(point, spec.flops);
      if (dtype_key.IsUniformDataType(PrimitiveType::BF16)) {
        std::array<int64_t, 4> fallback_point = {
            point[0],
            point[1] * 2,
            point[2] * 2,
            point[3] * 2,
        };
        fallback_interpolator->Add(fallback_point, spec.flops);
      }
    }
  }
  return std::unique_ptr<MatmulInterpolator>(new MatmulInterpolator(
      std::move(fallback_interpolator), std::move(interpolators)));
}

std::optional<absl::Duration> MatmulInterpolator::EstimatedRuntime(
    const HloInstruction& instr) const {
  std::optional<InterpolationSpecification> spec = GetInterpolationSpec(instr);
  if (!spec.has_value()) {
    VLOG(1) << "Unsupported instruction: " << instr.ToString();
    return std::nullopt;
  }

  MatmulDTypeKey key(spec->lhs_type, spec->rhs_type, spec->out_type);

  if (interpolators_ != nullptr && interpolators_->contains(key)) {
    return PreciseFlopsRetrieval(*spec, *interpolators_->at(key));
  }

  std::array<int64_t, 4> point = {
      spec->b,
      spec->m,
      spec->k,
      spec->n,
  };
  int64_t flops = 2ll * spec->b * spec->m * spec->k * spec->n;
  // NN interpolator is present. We use these performance tables as an override.
  if (nn_interpolator_ != nullptr) {
    return absl::Seconds(1.0 * flops / nn_interpolator_->Eval(point));
  }
  return absl::Seconds(1.0 * flops / fallback_interpolator_->Eval(point));
}

}  // namespace xla::gpu
