/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_INVARIANT_CODE_MOTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_INVARIANT_CODE_MOTION_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// HLO pass that rewrites while loops to hoist loop invariant instructions in
// the while body into the computation that contains the while instruction.

class WhileLoopInvariantCodeMotion : public HloPassInterface {
 public:
  ~WhileLoopInvariantCodeMotion() override = default;

  tensorflow::StringPiece name() const override {
    return "while-loop-invariant-code-motion";
  }
  StatusOr<bool> Run(HloModule* module) override;
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_WHILE_LOOP_INVARIANT_CODE_MOTION_H_
