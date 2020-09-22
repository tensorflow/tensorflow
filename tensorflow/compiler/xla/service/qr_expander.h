/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_QR_EXPANDER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_QR_EXPANDER_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/op_expander_pass.h"

namespace xla {

class QrExpander : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "qr_expander"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;

  struct QrResult {
    // The upper-triangular matrix R, packed together with the lower-triangular
    // elementary Householder reflectors `vs` below the diagonal.
    XlaOp a;

    // Representation of the Householder matrices I - beta v v.T
    XlaOp taus;  // Shape: [..., min(m, n)]
  };

  virtual StatusOr<QrResult> QrBlock(XlaOp a,
                                     PrecisionConfig::Precision precision);

  virtual StatusOr<XlaOp> CompactWYRepresentation(
      PrimitiveType type, absl::Span<const int64> batch_dims, XlaOp vs,
      XlaOp taus, int64 m, int64 n, PrecisionConfig::Precision precision);

 private:
  StatusOr<XlaOp> BuildQrDecomposition(XlaOp a, int64 block_size,
                                       PrecisionConfig::Precision precision);

  // Mapping from op signatures to existing computations.
  absl::flat_hash_map<string, HloComputation*> computation_cache_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_QR_EXPANDER_H_
