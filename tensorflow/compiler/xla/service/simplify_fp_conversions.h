/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SIMPLIFY_FP_CONVERSIONS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SIMPLIFY_FP_CONVERSIONS_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/core/platform/statusor.h"

namespace xla {

// Simplifies chains of floating-point conversions.
//
// The algebraic simplifier will remove convert pairs of the form `X -> Y -> X`,
// only when they are a no-op (e.g. `bf16 -> f32 -> bf16`). This passes does
// similar, but will simplify any chain of float conversions, possibly improving
// accuracy (e.g. `f32 -> bf16 -> f32` is removed).
class SimplifyFPConversions : public HloModulePass {
 public:
  absl::string_view name() const override { return "simplify-fp-conversions"; }

  StatusOr<bool> Run(HloModule* module) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SIMPLIFY_FP_CONVERSIONS_H_
