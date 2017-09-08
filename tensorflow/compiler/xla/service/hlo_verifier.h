/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VERIFIER_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VERIFIER_H_

#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// HLO pass that verifies invariants of HLO instructions for each computation in
// the module.
class HloVerifier : public HloPassInterface {
 public:
  explicit HloVerifier(const std::function<int64(const Shape&)>& shape_size_fn)
      : shape_size_fn_(shape_size_fn) {}
  ~HloVerifier() override = default;
  tensorflow::StringPiece name() const override { return "verifier"; }

  // Note: always returns false (no instructions are ever modified by this
  // pass).
  StatusOr<bool> Run(HloModule* module) override;

 private:
  // CHECKs various invariants of a fusion instruction.
  Status CheckFusionInstruction(HloInstruction* fusion) const;

  // Returns the size of a Shape in bytes.
  const std::function<int64(const Shape&)> shape_size_fn_;
};

}  // namespace xla

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VERIFIER_H_
