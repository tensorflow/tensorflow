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

// This file canonicalize reduction ops in hlo dialect to match the
// capacity of codegen backend.

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_HLOCANONICALIZEREDUCTIONPASS
#include "mlir-hlo/Dialect/mhlo/transforms/mhlo_passes.h.inc"

namespace {

// All the reduce ops can be divided into following four types:
//  - a) column reduction, only reduce the most significant dimensions.
//  - b) row reduction, only reduce the least significant dimensions.
//  - c) reduce to scalar, all dimensions are reduced.
//  - d) others. (not support now, maybe use transpose to canonicalize)
//
// Currently we do following canonicalization to match the capacity of codegen
// backend.
//
// For case a):
// ====================================================================================
//   we convert all column reduction to rank-2 column reduction.
//   For example, suppose we have:
//   ```
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
//       ...
//       %2 = "mhlo.reduce"(%arg0, ...) ({...})
//         {dimensions = dense<[0]> : tensor<1xi64>} :
//         (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
//       return %2 : tensor<?x?xf32>
//     }
//  ```
//   After conversion:
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
//       // [a, b, c] -> [a, b*c]
//       %1 = mhlo.dynamic_reshape(%arg0, ...) : (tensor<?x?x?xf32>,
//       tensor<2xi64>) -> tensor<?x?xf32> %2 = "mhlo.reduce"(%1, ...) ({...})
//         {dimensions = dense<[0]> : tensor<1xi64>} :
//         (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
//       %3 = "mhlo.dynamic_reshape"(%2, ...) : (tensor<?xf32>, tensor<1xi64>)
//       -> tensor<?x?f32> return %3 : tensor<?x?xf32>
//     }
//  ```
//
// For case b):
// ====================================================================================
//   we convert all row reduction to rank-2 row reduction.
//   For example, suppose we have:
//   ```
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
//       ...
//       %2 = "mhlo.reduce"(%arg0, ...) ({...})
//         {dimensions = dense<[2]> : tensor<1xi64>} :
//         (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
//       return %2 : tensor<?x?xf32>
//     }
//  ```
//   After conversion:
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
//       // [a, b, c] -> [a*b, c]
//       %1 = mhlo.dynamic_reshape(%arg0, ...) : (tensor<?x?x?xf32>,
//       tensor<2xi64>) -> tensor<?x?xf32> %2 = "mhlo.reduce"(%1, ...) ({...})
//         {dimensions = dense<[1]> : tensor<1xi64>} :
//         (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
//       %3 = "mhlo.dynamic_reshape"(%2, ...) : (tensor<?xf32>, tensor<1xi64>)
//       -> tensor<?x?f32> return %3 : tensor<?x?xf32>
//     }
//  ```
//
// For case c):
// ====================================================================================
//   we convert all reduce-to-scalar to rank-2 column reduction.
//
//   For example, suppose we have:
//   ```
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<f32> {
//       ...
//       %2 = "mhlo.reduce"(%arg0, ...) ({...})
//         {dimensions = dense<[0,1,2]> : tensor<3xi64>} :
//         (tensor<?x?x?xf32>, tensor<f32>) -> tensor<f32>
//       return %2 : tensor<f32>
//     }
//  ```
//   After conversion:
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<f32> {
//       // [a, b, c] -> [a*b*c, 1]
//       %1 = mhlo.dynamic_reshape(%arg0, ...) : (tensor<?x?x?xf32>,
//       tensor<2xi64>) -> tensor<?x?xf32> %2 = "mhlo.reduce"(%1, ...) ({...})
//         {dimensions = dense<[0]> : tensor<1xi64>} :
//         (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
//       %3 = "mhlo.reshape"(%2, ...) : (tensor<?xf32>, tensor<1xi64>) ->
//       tensor<f32> return %3 : tensor<f32>
//     }
//  ```

struct HloCanonicalizeReductionPass
    : impl::HloCanonicalizeReductionPassBase<HloCanonicalizeReductionPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<tensor::TensorDialect>();
  }
  void runOnOperation() override {
    getOperation().walk([&](ReduceOp op) {
      SmallVector<int64_t, 4> dimsToReduce;
      DenseSet<int64_t> dimsToReduceSet;
      for (auto dim : op.dimensions().getValues<APInt>()) {
        dimsToReduce.push_back(dim.getSExtValue());
        dimsToReduceSet.insert(dimsToReduce.back());
      }

      // empty reduction is just a no-op, thus no need to do codegen.
      if (dimsToReduce.empty()) return;

      // suppose reduce input is a ranked tensor
      auto ty = op.getOperand(0).getType().dyn_cast<RankedTensorType>();
      if (!ty) return signalPassFailure();
      int rank = ty.getRank();
      int ndimsToReduce = dimsToReduce.size();
      auto elemTy = ty.getElementType();
      llvm::sort(dimsToReduce);

      // skip case d) form since we don't support it.
      if ((dimsToReduce.back() - dimsToReduce[0]) != (ndimsToReduce - 1) ||
          (dimsToReduce[0] != 0 && dimsToReduce.back() != (rank - 1))) {
        return;
      }

      // rank 2 row/column reduction is already supported.
      if (rank == 2 && ndimsToReduce == 1) {
        return;
      }

      SmallVector<int64_t, 4> dimsToKeep;
      for (int i = 0; i < rank; ++i) {
        if (!dimsToReduceSet.count(i)) dimsToKeep.push_back(i);
      }

      OpBuilder b(op);
      auto loc = op.getLoc();
      // TODO(disc): uniformed shape_scalar_type with shape_derivation
      auto shapeScalarType = b.getIntegerType(32);
      auto one = b.create<arith::ConstantIntOp>(loc, 1ll, shapeScalarType);

      // funtion to get total elements in selected dimensions
      auto dimProd = [&](ArrayRef<int64_t> dims) {
        Value nelems = one;
        for (int64_t v : dims) {
          Value dimIndex = b.create<tensor::DimOp>(loc, op.getOperand(0), v);
          nelems = b.create<arith::MulIOp>(
              loc, nelems,
              b.create<arith::IndexCastOp>(loc, shapeScalarType, dimIndex));
        }
        return nelems;
      };

      SmallVector<Value, 2> newOperandDims;
      DenseIntElementsAttr attr;
      Value nelemToReduce = dimProd(dimsToReduce);
      Value nelemToKeep = dimProd(dimsToKeep);
      if (rank == ndimsToReduce) {
        // case c) Reduce to scalar.
        // Currently we don't support reduce to scalar directly.
        // As a workaround, we convert the `reduce to scalar` to a rank 2
        // column reduction having following form:
        // Suppose nelems = ProdutionOp(ShapeOp(I)), We convert I into
        // shape `[nelems, 1]`.
        // TODO(disc): this may have performance issue. Implements a reduce to
        // scalar schedule if necessary.
        newOperandDims.push_back(nelemToReduce);
        newOperandDims.push_back(nelemToKeep);
        attr = DenseIntElementsAttr::get(
            RankedTensorType::get({1}, b.getIntegerType(64)), {0ll});
      } else if (dimsToReduce[0] == 0) {
        // case a) column reduction
        newOperandDims.push_back(nelemToReduce);
        newOperandDims.push_back(nelemToKeep);
        attr = DenseIntElementsAttr::get(
            RankedTensorType::get({1}, b.getIntegerType(64)), {0ll});
      } else {
        // case b) row reduction
        newOperandDims.push_back(nelemToKeep);
        newOperandDims.push_back(nelemToReduce);
        attr = DenseIntElementsAttr::get(
            RankedTensorType::get({1}, b.getIntegerType(64)), {1ll});
      }

      Value newOperandShape =
          b.create<tensor::FromElementsOp>(loc, newOperandDims);

      SmallVector<Value, 4> newOperands;
      for (Value operand : op.operands()) {
        newOperands.push_back(b.create<DynamicReshapeOp>(
            loc,
            RankedTensorType::get(
                SmallVector<int64_t, 4>(newOperandDims.size(),
                                        ShapedType::kDynamicSize),
                elemTy),
            operand, newOperandShape));
      }
      auto newOp = b.create<ReduceOp>(loc, newOperands, op.init_values(), attr);
      newOp.body().takeBody(op.body());

      SmallVector<Value, 4> newResults;
      if (dimsToKeep.empty()) {
        // case c) reduce to scalar
        // reshape rank 1 tensor with size 1 to a rank 0 tensor
        for (Value result : newOp.getResults()) {
          newResults.push_back(b.create<ReshapeOp>(
              loc, RankedTensorType::get({}, elemTy), result));
        }
      } else {
        SmallVector<Value, 4> resultDims;
        for (int64_t i : dimsToKeep) {
          Value dimIndex = b.create<tensor::DimOp>(loc, op.getOperand(0), i);
          resultDims.push_back(
              b.create<arith::IndexCastOp>(loc, shapeScalarType, dimIndex));
        }
        Value resultShape = b.create<tensor::FromElementsOp>(loc, resultDims);
        for (auto&& e : llvm::zip(op.getResults(), newOp.getResults())) {
          newResults.push_back(b.create<DynamicReshapeOp>(
              loc, std::get<0>(e).getType(), std::get<1>(e), resultShape));
        }
      }
      for (auto&& e : llvm::zip(op.getResults(), newResults)) {
        std::get<0>(e).replaceAllUsesWith(std::get<1>(e));
      }
      op.erase();
    });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createHloCanonicalizeReductionPass() {
  return std::make_unique<HloCanonicalizeReductionPass>();
}

}  // namespace mhlo
}  // namespace mlir
