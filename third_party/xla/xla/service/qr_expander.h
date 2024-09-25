/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_QR_EXPANDER_H_
#define XLA_SERVICE_QR_EXPANDER_H_

#include "absl/container/flat_hash_map.h"
#include "xla/client/lib/qr.h"
#include "xla/client/xla_builder.h"
#include "xla/service/op_expander_pass.h"

namespace xla {

class QrExpander : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "qr_expander"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;

  virtual absl::StatusOr<QrDecomposition> QrBlock(
      XlaOp a, PrecisionConfig::Precision precision);

  virtual absl::StatusOr<XlaOp> CompactWYRepresentation(
      PrimitiveType type, absl::Span<const int64_t> batch_dims, XlaOp vs,
      XlaOp taus, int64_t m, int64_t n, PrecisionConfig::Precision precision);

 private:
  absl::StatusOr<XlaOp> BuildQrDecomposition(
      XlaOp a, int64_t block_size, PrecisionConfig::Precision precision);

  absl::StatusOr<XlaOp> ProductOfElementaryHouseholderReflectors(
      XlaOp a, XlaOp taus, int64_t block_size,
      PrecisionConfig::Precision precision);

  // Mapping from op signatures to existing computations.
  absl::flat_hash_map<std::string, HloComputation*> computation_cache_;
};

}  // namespace xla

#endif  // XLA_SERVICE_QR_EXPANDER_H_
