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

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "dnnl.hpp"  // NOLINT: for DNNL_MAX_NDIMS
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/dot_dims.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/cpu_info.h"

namespace xla::cpu {

bool IsOneDnnSupportedDType(PrimitiveType dtype) {
  using tsl::port::CPUFeature;
  switch (dtype) {
    case F32:
      return true;
    case BF16:
      return TestCPUFeature(CPUFeature::AVX512F) ||
             TestCPUFeature(CPUFeature::AVX_NE_CONVERT) ||
             TestCPUFeature(CPUFeature::AMX_BF16);
    case F16:
      return (TestCPUFeature(CPUFeature::AVX512BW) &&
              (TestCPUFeature(CPUFeature::AVX512_FP16) ||
               TestCPUFeature(CPUFeature::AMX_FP16))) ||
             TestCPUFeature(CPUFeature::AVX_NE_CONVERT);
    default:
      return false;
  }
}

bool IsOneDnnSupportedDType(PrimitiveType dtype,
                            const TargetMachineFeatures* cpu_features) {
  if (dtype == F32) {
    return true;
  }

  if (cpu_features == nullptr) {
    return IsOneDnnSupportedDType(dtype);
  }

  if (dtype == BF16) {
    return cpu_features->has_avx512bf16();
  }
  if (dtype == F16) {
    return cpu_features->has_avx512fp16();
  }

  return false;
}

absl::StatusOr<bool> IsOneDnnDotSupported(
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

}  // namespace xla::cpu
