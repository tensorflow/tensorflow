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

#include <iterator>
#include <memory>
#include <optional>
#include <utility>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORCLUSTERFUNCTIONCONVERSION
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

// Attach layouts for all the returned values so that custom device could get
// layouts for the handles.
mlir::LogicalResult AttachRetvalLayouts(
    mlir::OpBuilder* builder, mlir::TF::StatefulPartitionedCallOp sp_call_op) {
  // Find the FuncOp that the StatefulPartitionedCallOp is invoking.
  mlir::SymbolRefAttr sym =
      sp_call_op.getCallableForCallee().dyn_cast<mlir::SymbolRefAttr>();
  if (!sym)
    return sp_call_op.emitOpError(
        "has no symbolRef for given StatefulPartitionedCallOp");

  auto func = mlir::dyn_cast<mlir::func::FuncOp>(
      mlir::SymbolTable::lookupNearestSymbolFrom(sp_call_op, sym));
  if (!func)
    return sp_call_op.emitOpError() << "found no FuncOp for symbol " << sym;

  llvm::SmallVector<std::optional<Layout>, 8> retvals_layouts;
  retvals_layouts.reserve(func.getNumResults());
  for (auto operand : func.front().getTerminator()->getOperands()) {
    auto result_layout_or_status = ExtractLayoutFromOperand(operand);
    if (!result_layout_or_status.ok()) {
      return func.emitOpError("error while parsing result layout for function");
    }

    auto result_layout = result_layout_or_status.value();

    // When function returns its arguments directly, layout information for the
    // return value of `func` may be only obtainable by looking at it's callsite
    // operations. In that case, query the input layouts for function callsite
    // operations for layout information.
    if (!result_layout) {
      if (auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
        auto layout_or_status = ExtractLayoutFromOperand(
            sp_call_op.getOperand(block_arg.getArgNumber()));
        if (!layout_or_status.ok())
          return func.emitOpError(
              "error while parsing result layout for function");
        result_layout = std::move(layout_or_status.value());
      }

      if (!result_layout)
        return func.emitOpError(
            llvm::formatv("missing result layout attribute for function. All "
                          "DTensor functions "
                          "must have layouts for its results."));
    }
    retvals_layouts.emplace_back(result_layout.value());
  }

  // Note that we set this unconditionally - retvals_layout could be empty, but
  // that is fine and we will have an empty _layout for the
  // StatefulPartitionedCallOp. This is fine as for op without return values,
  // all we need is a placeholder layout so that no special case is needed in
  // dtensor_device.
  SetLayoutOnOp(sp_call_op,
                absl::Span<const absl::optional<Layout>>(
                    retvals_layouts.data(), retvals_layouts.size()));

  return mlir::success();
}

// Add an anotation to skip xla compilation for VarHandleOp and
// DestroyResourceOp.
void MaybeSkipXlaCompilation(mlir::OpBuilder* builder,
                             mlir::Operation* call_op) {
  auto function = MaybeFindFunction(call_op);
  const auto& body_ops = function->getBody().front().without_terminator();
  // VarHandleOp and DestroyResourceOp run on op-by-op mode, so there is only
  // one op in the function body.
  if (std::distance(std::begin(body_ops), std::end(body_ops)) == 1 &&
      llvm::isa<mlir::TF::VarHandleOp, mlir::TF::DestroyResourceOp>(
          body_ops.begin())) {
    call_op->setAttr(kSkipXlaCompilation, builder->getBoolAttr(true));
  }
}

mlir::LogicalResult ReplaceClusterWithPartitionCallOp(
    mlir::OpBuilder* builder, mlir::tf_device::ClusterFuncOp cluster_func) {
  auto mesh_attr = cluster_func->getAttrOfType<mlir::StringAttr>(kMeshAttr);
  if (!mesh_attr)
    return cluster_func.emitOpError()
           << "requires " << llvm::StringRef(kMeshAttr) << " attribute";

  llvm::SmallVector<mlir::Type, 8> output_types{
      cluster_func.getResultTypes().begin(),
      cluster_func.getResultTypes().end()};

  llvm::StringRef function_name = cluster_func.getFunc();

  builder->setInsertionPoint(cluster_func);
  auto call_op = builder->create<mlir::TF::StatefulPartitionedCallOp>(
      cluster_func.getLoc(), output_types, cluster_func.getOperands(),
      /*args_attrs=*/nullptr, /*res_attrs=*/nullptr, function_name, mesh_attr,
      /*config_proto=*/builder->getStringAttr(""),
      /*executor_type=*/builder->getStringAttr(""));

  MaybeSkipXlaCompilation(builder, call_op);

  if (mlir::failed(ValidateMetadataAttributes(cluster_func)))
    return mlir::failure();

  // All attributes beginning with `_` is validate, perform copy.
  mlir::TF::CopyUnderscoredAttributes(cluster_func, call_op);

  cluster_func.replaceAllUsesWith(call_op.getResults());
  cluster_func.erase();

  return AttachRetvalLayouts(builder, call_op);
}

// MLIR pass that converts tf_device.cluster_func to TF partitioned call
// op with device mesh config added to `config` attribute.
struct DTensorClusterFunctionConversion
    : public impl::DTensorClusterFunctionConversionBase<
          DTensorClusterFunctionConversion> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();

    // Find all tf_device.ClusterFunc ops and visit them in post order. This
    // order guarantees that ops in function definition is visited before
    // function call site operations. When python graph includes tf.functions
    // this leads to nested tf_device.ClusterFunc ops. As we infer the layout
    // of function call operations with layout attached to return values in the
    // function definition, ClusterFunc op in nested/inner functions must be
    // visited before ClusterFunc op in outer functions.
    llvm::SmallVector<mlir::tf_device::ClusterFuncOp, 8> clusters;
    getOperation().walk([&](mlir::tf_device::ClusterFuncOp cluster_func) {
      clusters.emplace_back(cluster_func);
    });

    mlir::OpBuilder op_builder(&context);
    for (auto cluster_func : llvm::reverse(clusters)) {
      if (mlir::failed(
              ReplaceClusterWithPartitionCallOp(&op_builder, cluster_func))) {
        return signalPassFailure();
      }
    }
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorClusterFunctionConversion() {
  return std::make_unique<DTensorClusterFunctionConversion>();
}

}  // namespace dtensor
}  // namespace tensorflow
