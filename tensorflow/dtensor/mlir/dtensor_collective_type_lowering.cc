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

#include <memory>
#include <optional>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {
namespace {

#define GEN_PASS_DEF_DTENSORCOLLECTIVETYPELOWERINGPASS
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

mlir::LogicalResult WrapOpWithCasts(const mlir::RankedTensorType& input_type,
                                    const mlir::RankedTensorType& output_type,
                                    mlir::Operation* reduce_op) {
  mlir::OpBuilder builder(reduce_op);
  auto intermediate_type = mlir::RankedTensorType::get(
      output_type.getShape(), input_type.getElementType());

  const mlir::Location loc = reduce_op->getLoc();
  mlir::TF::CastOp cast_to_long = builder.create<mlir::TF::CastOp>(
      loc, input_type, reduce_op->getOperand(0));
  reduce_op->setOperand(0, cast_to_long.getY());
  reduce_op->getResult(0).setType(intermediate_type);

  mlir::Value result = reduce_op->getResult(0);
  builder.setInsertionPointAfter(reduce_op);
  mlir::TF::CastOp cast_to_original =
      builder.create<mlir::TF::CastOp>(loc, output_type, result);
  StatusOr<Layout> result_layout =
      ExtractRequiredSingleLayoutFromOp(result.getDefiningOp());

  if (!result_layout.ok()) {
    return reduce_op->emitOpError(result_layout.status().message());
  }
  SetSingleLayoutOnOp(cast_to_original, *result_layout);
  reduce_op->getResult(0).replaceAllUsesExcept(cast_to_original.getY(),
                                               cast_to_original);
  return mlir::success();
}

template <class ReduceOpType>
mlir::LogicalResult ConvertShortIntReduce(ReduceOpType reduce_op) {
  mlir::OpBuilder builder(reduce_op);
  StatusOr<Layout> output_layout = ExtractRequiredSingleLayoutFromOp(reduce_op);
  if (!output_layout.ok()) {
    return reduce_op.emitOpError(output_layout.status().message());
  }
  const mlir::Type output_type = reduce_op.getResult().getType();
  const mlir::Type input_type = reduce_op.getOperand(0).getType();

  // Handle bools by first casting to int32 and swapping All/Any for Min/Max.
  const mlir::TensorType& tensor_input_type =
      mlir::dyn_cast<mlir::TensorType>(input_type);
  const mlir::TensorType& tensor_output_type =
      mlir::dyn_cast<mlir::TensorType>(output_type);
  if (!tensor_input_type) return mlir::success();
  if (!tensor_output_type) return mlir::success();

  if (tensor_input_type.getElementType().isInteger(1)) {
    if (reduce_op.getReduceOpAttr().getValue().str() == kReduceOpAll)
      reduce_op.setReduceOpAttr(
          builder.getStringAttr(std::string(kReduceOpMin)));
    else if (reduce_op.getReduceOpAttr().getValue().str() == kReduceOpAny)
      reduce_op.setReduceOpAttr(
          builder.getStringAttr(std::string(kReduceOpMax)));
    else if (reduce_op.getReduceOpAttr().getValue().str() != kReduceOpMax &&
             reduce_op.getReduceOpAttr().getValue().str() != kReduceOpMin)
      return reduce_op.emitOpError()
             << "reduce for boolean only supports 'All'/'Min' or 'Any'/'Max' "
                "reduction. "
             << "Received '" << reduce_op.getReduceOpAttr().getValue().str()
             << "'";
  }
  if (auto integer_type = mlir::dyn_cast<mlir::IntegerType>(
          tensor_input_type.getElementType())) {
    int32_t min_width = 64;
    if (output_layout->mesh().is_tpu_mesh()) {
      min_width = 32;
    }

    if (integer_type.getWidth() >= min_width) {
      return mlir::success();
    }
    auto input_type = mlir::RankedTensorType::get(
        tensor_input_type.getShape(), builder.getIntegerType(min_width));

    auto output_type = mlir::RankedTensorType::get(
        tensor_output_type.getShape(), integer_type);
    return WrapOpWithCasts(input_type, output_type, reduce_op);
  }
  if (mlir::isa<mlir::BFloat16Type>(tensor_input_type.getElementType())) {
    if (output_layout->mesh().is_tpu_mesh()) {
      return mlir::success();
    }
    auto input_type = mlir::RankedTensorType::get(tensor_input_type.getShape(),
                                                  builder.getF32Type());

    auto output_type = mlir::RankedTensorType::get(
        tensor_output_type.getShape(), tensor_input_type.getElementType());

    return WrapOpWithCasts(input_type, output_type, reduce_op);
  }
  return mlir::success();
}

// Complex for AllReduce and ReduceScatter
template <class ReduceOpType>
mlir::LogicalResult ConvertComplexReduce(ReduceOpType reduce_op) {
  ReduceOpType real_reduce_op;
  ReduceOpType imag_reduce_op;
  mlir::OpBuilder builder(reduce_op);
  StatusOr<Layout> output_layout = ExtractRequiredSingleLayoutFromOp(reduce_op);
  if (!output_layout.ok()) {
    return reduce_op.emitOpError(output_layout.status().message());
  }

  const mlir::Value tensor_input = reduce_op.getInput();
  const mlir::Value tensor_result = reduce_op.getResult();
  const mlir::TensorType complex_input_tensor_type =
      mlir::dyn_cast<mlir::TensorType>(tensor_input.getType());
  if (!complex_input_tensor_type) {
    return mlir::success();
  }
  const mlir::TensorType complex_result_tensor_type =
      mlir::dyn_cast<mlir::TensorType>(tensor_result.getType());
  if (!complex_result_tensor_type) {
    return mlir::success();
  }
  auto input_element_type = mlir::dyn_cast<mlir::ComplexType>(
      complex_input_tensor_type.getElementType());
  if (!input_element_type) {
    return mlir::success();
  }
  auto real_input_tensor_type =
      mlir::RankedTensorType::get(complex_input_tensor_type.getShape(),
                                  input_element_type.getElementType());
  auto real_result_tensor_type =
      mlir::RankedTensorType::get(complex_result_tensor_type.getShape(),
                                  input_element_type.getElementType());
  const mlir::Value tensor_temp_real = builder.create<mlir::TF::RealOp>(
      reduce_op.getLoc(), real_input_tensor_type, tensor_input);
  const mlir::Value tensor_temp_imag = builder.create<mlir::TF::ImagOp>(
      reduce_op.getLoc(), real_input_tensor_type, tensor_input);
  real_reduce_op = mlir::dyn_cast<ReduceOpType>(builder.clone(*reduce_op));
  real_reduce_op->setOperand(0, tensor_temp_real);
  real_reduce_op->getResult(0).setType(real_result_tensor_type);
  imag_reduce_op = mlir::dyn_cast<ReduceOpType>(builder.clone(*reduce_op));
  imag_reduce_op->setOperand(0, tensor_temp_imag);
  imag_reduce_op->getResult(0).setType(real_result_tensor_type);
  const mlir::Type output_type = reduce_op.getResult().getType();
  auto complex_reduce_op = builder.create<mlir::TF::ComplexOp>(
      reduce_op->getLoc(), output_type, real_reduce_op.getResult(),
      imag_reduce_op.getResult());
  StatusOr<Layout> desired_layout =
      ExtractRequiredSingleLayoutFromOp(reduce_op);
  SetSingleLayoutOnOp(complex_reduce_op, *desired_layout);
  reduce_op.getOutput().replaceAllUsesWith(complex_reduce_op.getResult());
  reduce_op.erase();
  return mlir::success();
}

// Complex for AllToAll, AllGather, and AllScatter
template <class CollectiveType>
mlir::LogicalResult ConvertComplexCollectives(CollectiveType op) {
  CollectiveType real_op;
  CollectiveType imag_op;
  mlir::OpBuilder builder(op);
  StatusOr<Layout> output_layout = ExtractRequiredSingleLayoutFromOp(op);
  if (!output_layout.ok()) {
    return op.emitOpError(output_layout.status().message());
  }

  const mlir::Value tensor_input = op.getInput();
  const mlir::Value tensor_result = op.getResult();
  const mlir::TensorType complex_input_tensor_type =
      mlir::dyn_cast<mlir::TensorType>(tensor_input.getType());
  if (!complex_input_tensor_type) {
    return mlir::success();
  }
  const mlir::TensorType& complex_result_tensor_type =
      mlir::dyn_cast<mlir::TensorType>(tensor_result.getType());
  if (!complex_result_tensor_type) {
    return mlir::success();
  }

  auto input_element_type = mlir::dyn_cast<mlir::ComplexType>(
      complex_input_tensor_type.getElementType());
  if (!input_element_type) {
    return mlir::success();
  }
  auto real_input_tensor_type =
      mlir::RankedTensorType::get(complex_input_tensor_type.getShape(),
                                  input_element_type.getElementType());
  auto real_result_tensor_type =
      mlir::RankedTensorType::get(complex_result_tensor_type.getShape(),
                                  input_element_type.getElementType());
  const mlir::Value tensor_temp_real = builder.create<mlir::TF::RealOp>(
      op.getLoc(), real_input_tensor_type, tensor_input);
  const mlir::Value tensor_temp_imag = builder.create<mlir::TF::ImagOp>(
      op.getLoc(), real_input_tensor_type, tensor_input);
  real_op = mlir::dyn_cast<CollectiveType>(builder.clone(*op));
  real_op->setOperand(0, tensor_temp_real);
  real_op->getResult(0).setType(real_result_tensor_type);
  imag_op = mlir::dyn_cast<CollectiveType>(builder.clone(*op));
  imag_op->setOperand(0, tensor_temp_imag);
  imag_op->getResult(0).setType(real_result_tensor_type);
  const mlir::Type output_type = op.getResult().getType();
  auto complex_op = builder.create<mlir::TF::ComplexOp>(
      op.getLoc(), output_type, real_op.getResult(), imag_op.getResult());
  const Layout desired_layout = op.getOutputLayout();
  SetSingleLayoutOnOp(complex_op, desired_layout);
  op.getOutput().replaceAllUsesWith(complex_op.getResult());
  op.erase();
  return mlir::success();
}

// A Walk that allows mutation inside parent.
template <typename FuncT, typename OpT = mlir::detail::first_argument<FuncT>>
mlir::LogicalResult MutatingWalk(mlir::Operation* parent, FuncT func) {
  llvm::SmallVector<OpT, 4> ops;
  parent->walk([&](OpT op) { ops.push_back(op); });
  for (auto op : ops) {
    if (mlir::failed(func(op))) {
      return mlir::LogicalResult::failure();
    }
  }
  return mlir::LogicalResult::success();
}

class DTensorCollectiveTypeLoweringPass
    : public impl::DTensorCollectiveTypeLoweringPassBase<
          DTensorCollectiveTypeLoweringPass> {
 public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    if (mlir::failed(
            MutatingWalk(func, [&](mlir::TF::DTensorAllReduceOp all_reduce) {
              // Lower integer type all reduce
              return ConvertComplexReduce(all_reduce);
            }))) {
      signalPassFailure();
    }

    if (mlir::failed(
            MutatingWalk(func, [&](mlir::TF::DTensorAllScatterOp all_scatter) {
              // Lower complex type all scatter
              return ConvertComplexCollectives(all_scatter);
            }))) {
      signalPassFailure();
    }

    if (mlir::failed(
            MutatingWalk(func, [&](mlir::TF::DTensorAllGatherOp all_gather) {
              // Lower complex type all gather.
              return ConvertComplexCollectives(all_gather);
            }))) {
      signalPassFailure();
    }

    if (mlir::failed(
            MutatingWalk(func, [&](mlir::TF::DTensorAllToAllOp all_to_all) {
              // Lower complex type all to all
              return ConvertComplexCollectives(all_to_all);
            }))) {
      signalPassFailure();
    }

    if (mlir::failed(MutatingWalk(
            func, [&](mlir::TF::DTensorReduceScatterOp reduce_scatter) {
              // Lower complex type reduce scatter.
              return ConvertComplexReduce(reduce_scatter);
            }))) {
      signalPassFailure();
    }

    if (mlir::failed(
            MutatingWalk(func, [&](mlir::TF::DTensorAllReduceOp all_reduce) {
              // Lower integer type all reduce
              return ConvertShortIntReduce(all_reduce);
            }))) {
      signalPassFailure();
    }

    if (mlir::failed(MutatingWalk(
            func, [&](mlir::TF::DTensorReduceScatterOp reduce_scatter) {
              // Lower integer type all reduce
              return ConvertShortIntReduce(reduce_scatter);
            }))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorCollectiveTypeLoweringPass() {
  return std::make_unique<DTensorCollectiveTypeLoweringPass>();
}

}  // namespace dtensor
}  // namespace tensorflow
