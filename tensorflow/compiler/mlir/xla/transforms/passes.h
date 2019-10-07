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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir

namespace mlir {

class FuncOp;
class Operation;
template <typename T>
class OpPassBase;

namespace xla_hlo {

/// Lowers from TF dialect to HLO dialect.
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeTFPass();

/// Converts the provided Operation as well as all nested operations into HLO
/// dialect using the conversion patterns registered by the HLO dialect.
LogicalResult legalizeTF(Operation* op);

/// Lowers HLO control flow ops to the Standard dialect.
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeControlFlowPass();

/// Lowers from HLO dialect to Standard dialect.
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeToStdPass();

// Lowers from HLO dialect to LHLO dialect allocating/deallocating temporary
// buffers if necessary.
//
// Example fusion with HLO ops.
//
// func @fusion(%arg0: memref<2x2xf32>,
//              %arg1: memref<2x2xf32>,
//              %arg2: memref<2x2xf32>,
//              %arg3: memref<2x2xf32>) {
//   "xla_lhlo.fusion"() ({
//     %0 = tensor_load %arg1 : memref<2x2xf32>
//     %1 = tensor_load %arg2 : memref<2x2xf32>
//     %2 = "xla_hlo.add"(%0, %1) {name = "add"} :
//         (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//     %3 = tensor_load %arg0 : memref<2x2xf32>
//     %4 = "xla_hlo.mul"(%2, %3) {name = "multiply"} :
//         (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
//     tensor_store %4, %arg3 : memref<2x2xf32>
//     "xla_lhlo.terminator"() : () -> ()
//   }) {name = "fusion"} : () -> ()
//   return
// }
//
// Transformed fusion with LHLO ops.
// func @fusion(%arg0: memref<2x2xf32>,
//              %arg1: memref<2x2xf32>,
//              %arg2: memref<2x2xf32>,
//              %arg3: memref<2x2xf32>) {
//   "xla_lhlo.fusion"() ( {
//     %0 = alloc() {temp = true} : memref<2x2xf32>
//     "xla_lhlo.add"(%arg1, %arg2, %0) :
//         (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
//     "xla_lhlo.mul"(%0, %arg0, %arg3) :
//         (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
//     dealloc %0 : memref<2x2xf32>
//     "xla_lhlo.terminator"() : () -> ()
//   }) {name = "fusion"} : () -> ()
//   return
//  }
// }
std::unique_ptr<OpPassBase<FuncOp>> createLegalizeToLhloPass();

}  // namespace xla_hlo

namespace xla_lhlo {

std::unique_ptr<OpPassBase<FuncOp>> createLegalizeToAffinePass();

std::unique_ptr<OpPassBase<FuncOp>> createLegalizeToLhloPass();

}  // namespace xla_lhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_PASSES_H_
