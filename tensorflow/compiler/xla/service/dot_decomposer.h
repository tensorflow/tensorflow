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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DOT_DECOMPOSER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DOT_DECOMPOSER_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// DotDecomposer is a pass which decomposes batch Dot operations into a
// sequence of smaller (R2) Dot operations.
class DotDecomposer : public HloPassInterface {
 public:
  // Decomposes batch Dot operations when 'decompose_batch_dot' is true.
  DotDecomposer(bool decompose_batch_dot = true)
      : decompose_batch_dot_(decompose_batch_dot) {}
  ~DotDecomposer() = default;
  tensorflow::StringPiece name() const override { return "dot_decomposer"; }

  // Run DotDecomposer pass on computations in 'module'.
  // Returns whether the 'module' was changed.
  StatusOr<bool> Run(HloModule* module) override;

 private:
  bool decompose_batch_dot_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DOT_DECOMPOSER_H_
