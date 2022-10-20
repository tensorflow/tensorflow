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

#ifndef MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_LINALG_UTILS_H
#define MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_LINALG_UTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace gml_st {

// Helper functions to match `linalg.generic` ops that implement simple
// reductions, bcasts, and cwise ops.

// Checks if an affine map maps all dimensions in sequence, skipping a unique
// dimension. This can be the output map of a reduction, or the input map of a
// bcast. For example:
//   - affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//   - affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
//   - affine_map<(d0, d1) -> (d0)>
//   - affine_map<(d0, d1) -> (d1)>
bool isBcastOrReductionMap(AffineMap map, int64_t &dim);

bool isSimpleReduction(Operation *op, int64_t &dim, Value &operand);

bool isCwiseGenericOp(Operation *op, int64_t &arity);

bool isUnaryCwiseGenericOp(Operation *op);

bool isSimpleBcast(Operation *op, int64_t &dim, Value &operand);

struct SimpleBcastReduction {
  Operation *bcast;
  Operation *reduction;
  Value operand;
};

bool isSimpleBcastReduction(Operation *op, int64_t &dim,
                            SimpleBcastReduction &chain);

}  // namespace gml_st
}  // namespace mlir

#endif