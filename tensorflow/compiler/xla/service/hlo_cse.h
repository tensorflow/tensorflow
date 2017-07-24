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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CSE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CSE_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which performs common-subexpression elimination. Identical constants
// and identical instructions with the same operands are commoned. The pass
// iterates over the instructions in topological order which enables the pass to
// find arbitrarily large common expressions.
class HloCSE : public HloPassInterface {
 public:
  // If is_layout_sensitive is true, then the simplifier preserves layout during
  // transformation. Otherwise, layout is ignored.
  explicit HloCSE(bool is_layout_sensitive)
      : is_layout_sensitive_(is_layout_sensitive) {}
  ~HloCSE() override {}
  tensorflow::StringPiece name() const override { return "cse"; }

  // Run CSE on the given module. Returns whether the module was changed (common
  // subexpressions were found and eliminated).
  StatusOr<bool> Run(HloModule* module) override;

 private:
  bool is_layout_sensitive_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_CSE_H_
