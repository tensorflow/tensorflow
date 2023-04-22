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
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace mhlo {
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
//       %2 = "mhlo.reduce"(%arg0, ...) ( {...})
//         {dimensions = dense<[0]> : tensor<1xi64>} :
//         (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
//       return %2 : tensor<?x?xf32>
//     }
//  ```
//   After conversion:
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
//       // [a, b, c] -> [a, b*c]
//       %1 = mhlo.dynamic_reshape(%arg0, ...) : (tensor<?x?x?xf32>,
//       tensor<2xi64>) -> tensor<?x?xf32> %2 = "mhlo.reduce"(%1, ...) ( {...})
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
//       %2 = "mhlo.reduce"(%arg0, ...) ( {...})
//         {dimensions = dense<[2]> : tensor<1xi64>} :
//         (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
//       return %2 : tensor<?x?xf32>
//     }
//  ```
//   After conversion:
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
//       // [a, b, c] -> [a*b, c]
//       %1 = mhlo.dynamic_reshape(%arg0, ...) : (tensor<?x?x?xf32>,
//       tensor<2xi64>) -> tensor<?x?xf32> %2 = "mhlo.reduce"(%1, ...) ( {...})
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
//       %2 = "mhlo.reduce"(%arg0, ...) ( {...})
//         {dimensions = dense<[0,1,2]> : tensor<3xi64>} :
//         (tensor<?x?x?xf32>, tensor<f32>) -> tensor<f32>
//       return %2 : tensor<f32>
//     }
//  ```
//   After conversion:
//     func @test(%arg0: tensor<?x?x?xf32>) -> tensor<f32> {
//       // [a, b, c] -> [a*b*c, 1]
//       %1 = mhlo.dynamic_reshape(%arg0, ...) : (tensor<?x?x?xf32>,
//       tensor<2xi64>) -> tensor<?x?xf32> %2 = "mhlo.reduce"(%1, ...) ( {...})
//         {dimensions = dense<[0]> : tensor<1xi64>} :
//         (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
//       %3 = "mhlo.reshape"(%2, ...) : (tensor<?xf32>, tensor<1xi64>) ->
//       tensor<f32> return %3 : tensor<f32>
//     }
//  ```

struct HloCanonicalizeReductionPass
    : HloCanonicalizeReductionPassBase<HloCanonicalizeReductionPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<tensor::TensorDialect>();
  }
  void runOnFunction() override {
    getFunction().walk([&](ReduceOp op) {
      SmallVector<int64_t, 4> dims_to_reduce;
      DenseSet<int64_t> dims_to_reduce_set;
      for (auto dim : op.dimensions().getIntValues()) {
        dims_to_reduce.push_back(dim.getSExtValue());
        dims_to_reduce_set.insert(dims_to_reduce.back());
      }

      // empty reduction is just a no-op, thus no need to do codegen.
      if (dims_to_reduce.empty()) return;

      // suppose reduce input is a ranked tensor
      auto ty = op.getOperand(0).getType().dyn_cast<RankedTensorType>();
      if (!ty) return signalPassFailure();
      int rank = ty.getRank();
      int ndims_to_reduce = dims_to_reduce.size();
      auto elem_ty = ty.getElementType();
      llvm::sort(dims_to_reduce);

      // skip case d) form since we don't support it.
      if ((dims_to_reduce.back() - dims_to_reduce[0]) !=
              (ndims_to_reduce - 1) ||
          (dims_to_reduce[0] != 0 && dims_to_reduce.back() != (rank - 1))) {
        return;
      }

      // rank 2 row/column reduction is already supported.
      if (rank == 2 && ndims_to_reduce == 1) {
        return;
      }

      SmallVector<int64_t, 4> dims_to_keep;
      for (int i = 0; i < rank; ++i) {
        if (!dims_to_reduce_set.count(i)) dims_to_keep.push_back(i);
      }

      OpBuilder b(op);
      auto loc = op.getLoc();
      // TODO(disc): uniformed shape_scalar_type with shape_derivation
      auto shape_scalar_type = b.getIntegerType(32);
      auto one = b.create<ConstantIntOp>(loc, 1ll, shape_scalar_type);

      // funtion to get total elements in selected dimensions
      auto dim_prod = [&](ArrayRef<int64_t> dims) {
        Value nelems = one;
        for (int64_t v : dims) {
          Value dim_index = b.create<tensor::DimOp>(loc, op.getOperand(0), v);
          nelems = b.create<MulIOp>(
              loc, nelems,
              b.create<IndexCastOp>(loc, dim_index, shape_scalar_type));
        }
        return nelems;
      };

      SmallVector<Value, 2> new_operand_dims;
      DenseIntElementsAttr attr;
      Value nelem_to_reduce = dim_prod(dims_to_reduce);
      Value nelem_to_keep = dim_prod(dims_to_keep);
      if (rank == ndims_to_reduce) {
        // case c) Reduce to scalar.
        // Currently we don't support reduce to scalar directly.
        // As a workaround, we convert the `reduce to scalar` to a rank 2
        // column reduction having following form:
        // Suppose nelems = ProdutionOp(ShapeOp(I)), We convert I into
        // shape `[nelems, 1]`.
        // TODO(disc): this may have performance issue. Implements a reduce to
        // scalar schedule if necessary.
        new_operand_dims.push_back(nelem_to_reduce);
        new_operand_dims.push_back(nelem_to_keep);
        attr = DenseIntElementsAttr::get(
            RankedTensorType::get({1}, b.getIntegerType(64)), {0ll});
      } else if (dims_to_reduce[0] == 0) {
        // case a) column reduction
        new_operand_dims.push_back(nelem_to_reduce);
        new_operand_dims.push_back(nelem_to_keep);
        attr = DenseIntElementsAttr::get(
            RankedTensorType::get({1}, b.getIntegerType(64)), {0ll});
      } else {
        // case b) row reduction
        new_operand_dims.push_back(nelem_to_keep);
        new_operand_dims.push_back(nelem_to_reduce);
        attr = DenseIntElementsAttr::get(
            RankedTensorType::get({1}, b.getIntegerType(64)), {1ll});
      }

      Value new_operand_shape =
          b.create<tensor::FromElementsOp>(loc, new_operand_dims);

      SmallVector<Value, 4> new_operands;
      for (Value operand : op.inputs()) {
        new_operands.push_back(b.create<DynamicReshapeOp>(
            loc,
            RankedTensorType::get(
                SmallVector<int64_t, 4>(new_operand_dims.size(),
                                        ShapedType::kDynamicSize),
                elem_ty),
            operand, new_operand_shape));
      }
      auto new_op =
          b.create<ReduceOp>(loc, new_operands, op.init_values(), attr);
      new_op.body().takeBody(op.body());

      SmallVector<Value, 4> new_results;
      if (dims_to_keep.empty()) {
        // case c) reduce to scalar
        // reshape rank 1 tensor with size 1 to a rank 0 tensor
        for (Value result : new_op.getResults()) {
          new_results.push_back(b.create<ReshapeOp>(
              loc, RankedTensorType::get({}, elem_ty), result));
        }
      } else {
        SmallVector<Value, 4> result_dims;
        for (int64_t i : dims_to_keep) {
          Value dim_index = b.create<tensor::DimOp>(loc, op.getOperand(0), i);
          result_dims.push_back(
              b.create<IndexCastOp>(loc, dim_index, shape_scalar_type));
        }
        Value result_shape = b.create<tensor::FromElementsOp>(loc, result_dims);
        for (auto&& e : llvm::zip(op.getResults(), new_op.getResults())) {
          new_results.push_back(b.create<DynamicReshapeOp>(
              loc, std::get<0>(e).getType(), std::get<1>(e), result_shape));
        }
      }
      for (auto&& e : llvm::zip(op.getResults(), new_results)) {
        std::get<0>(e).replaceAllUsesWith(std::get<1>(e));
      }
      op.erase();
    });
  }
};

}  // namespace

std::unique_ptr<FunctionPass> createHloCanonicalizeReductionPass() {
  return std::make_unique<HloCanonicalizeReductionPass>();
}

}  // namespace mhlo
}  // namespace mlir
