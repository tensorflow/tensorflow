/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef MLIR_HLO_DEALLOCATION_TRANSFORMS_ANALYSIS_H
#define MLIR_HLO_DEALLOCATION_TRANSFORMS_ANALYSIS_H

#include "deallocation/utils/util.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace deallocation {

class DeallocationAnalysis {
 public:
  // Returns the set of all possible values that may back the given value. A
  // value `A` is considered to back another value `B` if
  // a) `A` is an alloc or a bbarg
  // b) `B` depends on `A` (possibly indirectly)
  //
  // For example, in this IR:
  //
  // func.func @foo(%arg0: memref<i32>) -> memref<i32> {
  //   %c0 = arith.constant 0 : index
  //   %c4 = arith.constant 4 : index
  //   %c1 = arith.constant 1 : index
  //   %ret = scf.for %i = %c0 to %c4 step %c1 iter_args(%x = %arg0)
  //        -> memref<i32> {
  //     %y = some.op(%x) : memref<i32> -> memref<i32>
  //     scf.yield %y : memref<i32>
  //   }
  //   func.return %ret : memref<i32>
  // }
  //
  // `getBackingMemory(%ret)` is {`%arg0`, `%x`, `%y`}.
  const breaks_if_you_move_ops::ValueSet& getBackingMemory(Value source);

 private:
  void collectBackingMemory(Value source, DenseSet<Value>& visited,
                            breaks_if_you_move_ops::ValueSet& results);

  DenseMap<Value, breaks_if_you_move_ops::ValueSet> backingMemory;
};

}  // namespace deallocation
}  // namespace mlir

#endif  // MLIR_HLO_DEALLOCATION_TRANSFORMS_ANALYSIS_H
