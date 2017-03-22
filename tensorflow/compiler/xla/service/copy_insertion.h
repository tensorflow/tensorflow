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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_COPY_INSERTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_COPY_INSERTION_H_

#include "tensorflow/compiler/xla/service/buffer_liveness.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// HLO pass which inserts a copy of the root instruction (creating a new root)
// if the root is or points-to any constant or parameter instruction.
// If the root instruction is a Tuple, only tuple elements which point to
// constant or parameter instructions will be copied.
// Copy insertion is necessary because constant and parameter arrays have
// different lifetimes than computation results.
class CopyInsertion : public HloPassInterface {
 public:
  explicit CopyInsertion(bool copy_param_and_const = true)
      : copy_param_and_const_(copy_param_and_const) {}
  ~CopyInsertion() override {}
  tensorflow::StringPiece name() const override { return "copy-insertion"; }

  // Run the pass on the given module. Returns whether the module was changed
  // (copies were inserted).
  StatusOr<bool> Run(HloModule* module) override;

 protected:
  // Returns a copy of `hlo`. Looks in inserted_copies_ first to avoid making
  // duplicate copies.
  StatusOr<HloInstruction*> FindOrInsertCopy(HloInstruction* hlo);

  // Determines whether to insert copies if the root instruction is, or
  // points-to, any constant or parameter instruction.
  const bool copy_param_and_const_;

  // A map containing all copies inserted during the copy insertion pass. The
  // key is the copied instruction and the value is the copy.
  std::unordered_map<HloInstruction*, HloInstruction*> inserted_copies_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_COPY_INSERTION_H_
