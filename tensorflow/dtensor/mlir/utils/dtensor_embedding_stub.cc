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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "tensorflow/dtensor/mlir/create_dtensor_mlir_passes.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSOREMBEDDING
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

struct DTensorEmbedding : public impl::DTensorEmbeddingBase<DTensorEmbedding> {
  void runOnOperation() override {}
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorEmbeddingPass() {
  return std::make_unique<DTensorEmbedding>();
}

}  // namespace dtensor
}  // namespace tensorflow
