/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file defines the op traits used in the MLIR TensorFlow Lite dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_TRAITS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_TRAITS_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"  // TF:llvm-project

namespace mlir {
namespace OpTrait {
namespace TFL {
// The trait to specify the channel dimension index of the input (first operand)
// of an affine TFL op (Conv2D, DepthwiseConv2D, FullyConnected).
//
//   class Conv2DOp
//       : public Op<Conv2DOp, OpTrait::TFL::ChannelDimIndex<0>::Impl> {
//
template <int Index>
class ChannelDimIndex {
 public:
  template <typename ConcreteType>
  class Impl : public TraitBase<ConcreteType, ChannelDimIndex<Index>::Impl> {
   public:
    static int GetChannelDimIndex() { return Index; }
  };
};

}  // namespace TFL
}  // namespace OpTrait
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_TRAITS_H_
