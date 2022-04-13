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

#ifndef TENSORFLOW_CORE_TRANSFORMS_UTILS_UTILS_H_
#define TENSORFLOW_CORE_TRANSFORMS_UTILS_UTILS_H_

#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"

namespace mlir {

class Operation;
class NamedAttrList;

namespace tfg {
namespace util {

// Returns if the requested device is CPU.
bool NodeIsOnCpu(Operation *op);

// Erase the attribute starts with "_".
void EraseRegularNodeAttributes(NamedAttrList &attr_list);

// When rewriting an operation 1-to-1, intrinsic attributes are manually
// forwarded, modified, or dropped. For example, when `If` is rewritten to
// `IfRegion`,
//
// 1. `Tout` is forwarded as is,
// 2. `then_branch` is changed to `then_attrs` which contain the attribute
// dictionary part of the `#tf_type.func`, and
// 3. `Tin` is dropped.
//
// Non-intrinsic attributes, e.g. `_tpu_cluster`, are blindly forwarded to the
// new operation.
void ForwardNonIntrinsicAttributes(Operation *src, Operation *dst);

// Add an argument to a loop region. This inserts the new data argument and
// control argument at the correct positions and returns them. Also, this
// function updates any preserved argument attributes by inserting a null.
struct LoopRegionArgumentUpdate {
  BlockArgument data, ctl;
};
LoopRegionArgumentUpdate LoopRegionAddArgument(Region &region, Type type);

// Erase an argument from a loop region. This erases the corresponding control
// argument. Also, this function updates any preserved argument attributes by
// deleting them.
void LoopRegionEraseArgument(Region &region, unsigned index);

// Indicate that a result has been added to a loop region. Call this function to
// update the preserved result attributes.
void LoopRegionResultAdded(Region &region, unsigned num = 1);

// Indicate that a result has been erased from a loop region. Call this function
// to update the preserved result attributes.
void LoopRegionResultErased(Region &region, unsigned index);

// Erase operands from an op that might have an `operand_segment_sizes` ,
// updating the attribute in-place if present.
void SizedOperandSegmentsEraseOperands(Operation *op,
                                       ArrayRef<unsigned> indices);
void SizedOperandSegmentsEraseOperands(Operation *op,
                                       const llvm::BitVector &erase);

}  // namespace util
}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_TRANSFORMS_UTILS_UTILS_H_
