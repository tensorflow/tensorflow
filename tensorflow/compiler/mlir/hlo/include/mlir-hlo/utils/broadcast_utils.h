/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_UTILS_BROADCAST_UTILS_H
#define MLIR_HLO_UTILS_BROADCAST_UTILS_H

// Utilities relating to implementing HLO broadcasting.
// Note: This file should not depend on any non-MLIR TensorFlow libraries.

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace hlo {

// Checks whether the given operand types and broadcast_dims attr represent a
// legal combination for "numpy" style broadcasting (where 1-dims are prepended
// to the smaller ranked operand until it is of the same rank as the larger).
// See: https://docs.scipy.org/doc/numpy/reference/ufuncs.html
bool IsLegalNumpyRankedBroadcast(Value lhs, Value rhs,
                                 DenseIntElementsAttr broadcast_dims);

// Emits shape dialect ops to compute the result shape for a broadcasting
// binary/n-ary elementwise op which broadcasts according to "numpy" semantics
// (see above), returning an extent tensor of the resulting shape. The function
// should only be used in contexts that ensure both operands to be
// broadcastable.
Value ComputeBinaryElementwiseBroadcastingResultExtents(Location loc, Value lhs,
                                                        Value rhs,
                                                        OpBuilder& builder);
Value ComputeNaryElementwiseBroadcastingResultExtents(Location loc,
                                                      ValueRange operands,
                                                      OpBuilder& builder);

}  // namespace hlo
}  // namespace mlir

#endif  // MLIR_HLO_UTILS_BROADCAST_UTILS_H
