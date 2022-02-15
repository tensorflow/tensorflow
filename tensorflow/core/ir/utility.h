/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_IR_UTILITY_H_
#define TENSORFLOW_CORE_IR_UTILITY_H_

#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace tfg {

// Region-based loop ops store control tokens all after the data values, unlike
// functions which store them as pairs. This is required by
// RegionBranchOpInterface's API which requires MutableOperandRange, i.e. the
// data operands need to be stored contiguously.

// Given a region belonging to a region-based loop operation (e.g. a while
// loop), return the subrange of block arguments that are data values.
Block::BlockArgListType GetLoopRegionDataArgs(Region &region);
// Given a region belonging to a region-based loop operation (e.g. a while
// loop), return the subrange of block arguments that are control tokens.
Block::BlockArgListType GetLoopRegionControlTokens(Region &region);
// Given a data value block argument of a region belonging to a region-based
// loop operation (e.g. a while loop), return the block argument that
// corresponds to the control token.
BlockArgument GetLoopRegionControlOf(BlockArgument data);
// Given a control token block argument of a region belonging to a region-based
// loop operation (e.g. a while loop), return the block argument that
// corresponds to the data value.
BlockArgument GetLoopRegionDataOf(BlockArgument ctl);

// Given a TFG value, lookup the associated control token. For op results, the
// token will be the last result of the op. For block arguments, the token will
// be the subsequent argument. A data value always has an associated control
// token.
Value LookupControlDependency(Value data);

// Given a TFG control token, lookup the associated data value. Block arguments
// will always have an associated data value: the previous argument. For ops,
// if the only result is a control token, return None. Otherwise, returns the
// first result.
Optional<Value> LookupDataValue(Value ctl);

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_UTILITY_H_
