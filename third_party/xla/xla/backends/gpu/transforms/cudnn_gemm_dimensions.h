/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_GEMM_DIMENSIONS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_GEMM_DIMENSIONS_H_

#include <cstdint>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla::gpu {

// Sizes and strides of the tensors of a cuDNN GEMM fusion in canonical order:
// [batch, M, K] (LHS), [batch, K, N] (RHS) and [batch, M, N] (output).
class CuDnnGemmDimensions {
 public:
  // An operand of the dot (by operand number) or its output.
  enum class Port : int8_t { kLhs, kRhs, kLhsScale, kRhsScale, kOutput };

  // Analyzes the fusion of `dot`. Fails if cuDNN GEMMs cannot represent it.
  static absl::StatusOr<CuDnnGemmDimensions> Create(const HloInstruction& dot);

  CuDnnGemmDimensions(const CuDnnGemmDimensions&) = default;
  CuDnnGemmDimensions& operator=(const CuDnnGemmDimensions&) = default;
  CuDnnGemmDimensions(CuDnnGemmDimensions&&) = default;
  CuDnnGemmDimensions& operator=(CuDnnGemmDimensions&&) = default;

  // Layout of a parameter or of the output, or nullptr for other instructions.
  const CudnnTensorLayout* absl_nullable TensorLayout(
      const HloInstruction& hlo) const;

  // Canonical sizes of the port that `hlo` contributes to, or nullptr.
  const std::vector<int64_t>* absl_nullable CanonicalSizes(
      const HloInstruction& hlo) const;

 private:
  CuDnnGemmDimensions() = default;

  // Layouts of the parameters and of the output.
  absl::flat_hash_map<const HloInstruction*, CudnnTensorLayout> tensor_layouts_;
  // Port that each instruction of the cuDNN graph contributes to.
  absl::flat_hash_map<const HloInstruction*, Port> ports_;
  // Canonical sizes of each port.
  absl::flat_hash_map<Port, std::vector<int64_t>> canonical_sizes_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_CUDNN_GEMM_DIMENSIONS_H_
