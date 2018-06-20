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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_TUPLE_SIMPLIFIER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_TUPLE_SIMPLIFIER_H_

#include <utility>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which simplifies patterns of Tuple and GetTupleElement instructions in
// the module.
class TupleSimplifier : public HloPassInterface {
 public:
  TupleSimplifier() : TupleSimplifier(/*exclude_entry_computation=*/false) {}
  explicit TupleSimplifier(bool exclude_entry_computation);
  ~TupleSimplifier() override {}
  tensorflow::StringPiece name() const override { return "tuple-simplifier"; }

  // Run tuple simplification on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override;

 private:
  // When set, this pipeline stage will perform optimization of all computations
  // apart from the module's entry computation. This is used by Graphcore's
  // backend.
  bool exclude_entry_computation_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_TUPLE_SIMPLIFIER_H_
