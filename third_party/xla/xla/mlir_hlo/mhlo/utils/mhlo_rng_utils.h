/* Copyright 2023 The OpenXLA Authors.

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

#ifndef MLIR_HLO_MHLO_UTILS_MHLO_RNG_UTILS_H_
#define MLIR_HLO_MHLO_UTILS_MHLO_RNG_UTILS_H_

#include "mhlo/IR/hlo_ops.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace mhlo {

LogicalResult generateLinalgThreeFry(OpBuilder& builder, Location loc,
                                     ShapedType resultTy, Value& state,
                                     Value& result);

LogicalResult generateLinalgPhilox(OpBuilder& builder, Location loc,
                                   ShapedType resultTy, Value& state,
                                   Value& result);

}  // namespace mhlo
}  // namespace mlir

#endif  // MLIR_HLO_MHLO_UTILS_MHLO_RNG_UTILS_H_
