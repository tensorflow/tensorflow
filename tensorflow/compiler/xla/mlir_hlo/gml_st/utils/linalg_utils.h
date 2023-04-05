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

#ifndef MLIR_HLO_GML_ST_UTILS_LINALG_UTILS_H
#define MLIR_HLO_GML_ST_UTILS_LINALG_UTILS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
namespace gml_st {

// Helper functions to match Linalg ops that implement simple reductions,
// bcasts, and cwise ops.

struct SimpleBcastReduction {
  Operation *bcast;
  Operation *reduction;
  Value operand;
};

bool isSimpleBcastReduction(Operation *op, int64_t *dimension = nullptr,
                            SimpleBcastReduction *chain = nullptr);

}  // namespace gml_st
}  // namespace mlir

#endif
