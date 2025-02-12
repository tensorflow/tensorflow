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

#include "tensorflow/dtensor/mlir/expansions/io_op_spmd_expander.h"

#include <algorithm>
#include <vector>

#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/device_utils.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {

template <typename T>
StatusOr<mlir::Operation*> Expand(mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(const std::vector<Layout> output_layouts,
                      ExtractRequiredLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(const std::vector<Layout> operand_layouts,
                      ExtractRequiredLayoutFromOperands(op));
  if (!AllReplicated(output_layouts) || !AllReplicated(operand_layouts)) {
    return errors::Unimplemented(
        llvm::formatv("Expecting {0} to have input and output layouts to be "
                      "fully replicated but was not. ",
                      OpName(op))
            .str());
  }

  // Build an if op that only runs the op on device 0. Every other device
  // will run a no-op.
  mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
  mlir::SymbolTable symbol_table(module);
  mlir::Location location = op->getLoc();
  mlir::OpBuilder builder(op);

  auto func_type =
      mlir::FunctionType::get(builder.getContext(), op->getOperandTypes(),
                              llvm::ArrayRef<mlir::Type>{});
  // Build then_func that is the branch of device_id != 0, which only contains a
  // single NoOp.
  mlir::func::FuncOp then_func = mlir::func::FuncOp::create(
      location,
      llvm::formatv("{0}_then_func_{1}", OpName(op), OpHash(op)).str(),
      func_type, llvm::ArrayRef<mlir::NamedAttribute>{});
  // Set function visibility to private to indicate that it is only used in
  // this module.
  then_func.setVisibility(mlir::SymbolTable::Visibility::Private);
  mlir::Block* then_fn_block = then_func.addEntryBlock();
  mlir::OpBuilder then_fn_builder =
      mlir::OpBuilder::atBlockBegin(then_fn_block);
  then_fn_builder.create<mlir::TF::NoOp>(location);
  then_fn_builder.create<mlir::func::ReturnOp>(location);

  // Build else_func that is the branch of device_id == 0.
  // The else func is just the original op.
  mlir::func::FuncOp else_func = mlir::func::FuncOp::create(
      location,
      llvm::formatv("{0}_else_func_{1}", OpName(op), OpHash(op)).str(),
      func_type, llvm::ArrayRef<mlir::NamedAttribute>{});
  // Set function visibility to private to indicate that it is only used in
  // this module.
  else_func.setVisibility(mlir::SymbolTable::Visibility::Private);

  mlir::Block* else_fn_block = else_func.addEntryBlock();
  mlir::OpBuilder else_fn_builder =
      mlir::OpBuilder::atBlockBegin(else_fn_block);

  else_fn_builder.create<T>(location, op->getResultTypes(),
                            else_fn_block->getArguments());
  else_fn_builder.create<mlir::func::ReturnOp>(location);

  symbol_table.insert(then_func);
  symbol_table.insert(else_func);

  TF_ASSIGN_OR_RETURN(mlir::Value device_id, DeviceId(op));

  TF_ASSIGN_OR_RETURN(
      mlir::Value zero_scalar,
      CreateZeroScalarConst(
          builder, location,
          mlir::cast<mlir::TensorType>(device_id.getType()).getElementType()));

  mlir::TF::NotEqualOp not_equal = builder.create<mlir::TF::NotEqualOp>(
      location, device_id, zero_scalar,
      /*incompatible_shape_error=*/builder.getBoolAttr(false));

  mlir::Operation* if_op = builder.create<mlir::TF::IfOp>(
      location, then_func.getFunctionType().getResults(),
      /*cond=*/not_equal.getResult(),
      /*input=*/op->getOperands(),
      /*then_branch=*/then_func.getSymName(),
      /*else_branch=*/else_func.getSymName(), /*is_stateless=*/false);

  op->replaceAllUsesWith(if_op);
  op->erase();
  return if_op;
}

}  // namespace
StatusOr<mlir::Operation*> IOOpSPMDExpander::ExpandOp(mlir::Operation* op) {
  if (llvm::isa<mlir::TF::WriteSummaryOp>(op)) {
    return Expand<mlir::TF::WriteSummaryOp>(op);
  } else if (llvm::isa<mlir::TF::FlushSummaryWriterOp>(op)) {
    return Expand<mlir::TF::FlushSummaryWriterOp>(op);
  }
  return errors::Unimplemented(
      llvm::formatv("SPMD for op : {0} is not implemented ", OpName(op)).str());
}

// Always return a set of replicated layouts for now. If there is a case where
// a dtensor user is writing a large tensor that is sharded, then, we can
// support that in the future.
StatusOr<llvm::DenseMap<int, Layout>> IOOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> output_layouts(op->getNumResults());
  for (int i = 0; i < op->getNumResults(); ++i) {
    output_layouts[i] =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getResult(i)));
  }
  return output_layouts;
}

// Always return a set of replicated layouts. IO ops usually either have
// no output or a scalar output, in which case it is replicated.
StatusOr<llvm::DenseMap<int, Layout>> IOOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> input_layouts(op->getNumOperands());
  for (int i = 0; i < op->getNumOperands(); ++i) {
    int rank = std::max(0, ValueRank(op->getOperand(i)));
    input_layouts[i] = Layout::ReplicatedOnMesh(mesh, rank);
  }
  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
