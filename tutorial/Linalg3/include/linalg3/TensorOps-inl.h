//===- TensorOps-inl.h - Linalg dialect TensorOps operation implementation ===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

/// The TensorOp-inl.h inclusion pattern is chosen to allow gradual extension of
/// TensorOps by adding implementations as they are needed in the appropriate
/// step in the tutorial.
#ifndef LINALG3_TENSOROPS_INL_H_
#define LINALG3_TENSOROPS_INL_H_

#include "linalg1/Common.h"
#include "linalg2/TensorOps.h"

namespace linalg {

template <class ConcreteOp>
mlir::Value *
linalg::TensorContractionBase<ConcreteOp>::getInputView(unsigned i) {
  return *(getInputs().begin() + i);
}

template <class ConcreteOp>
mlir::Value *
linalg::TensorContractionBase<ConcreteOp>::getOutputView(unsigned i) {
  return *(getOutputs().begin() + i);
}

} // namespace linalg

#endif // LINALG3_TENSOROPS-INL_H_
