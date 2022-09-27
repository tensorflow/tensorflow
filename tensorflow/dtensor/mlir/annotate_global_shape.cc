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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORANNOTATEGLOBALSHAPE
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

// Sets `_global_shape` attributes to argument/return values of `function`.
void AnnotateFunctionArgRetvalGlobalShapes(mlir::func::FuncOp function,
                                           mlir::OpBuilder* builder) {
  for (const auto& argument_type_and_index :
       llvm::enumerate(function.getArgumentTypes())) {
    const int index = argument_type_and_index.index();
    const auto& argument_type = argument_type_and_index.value();
    // Extract TensorType from element of resource type to allow setting proper
    // global shape of resource types.
    if (auto resource_type = mlir::getElementTypeOrSelf(argument_type)
                                 .dyn_cast<mlir::TF::ResourceType>()) {
      auto subtype = resource_type.getSubtypes();
      if (subtype.size() == 1) {
        // subtype returns a Array of TensorType -- if it contains more than one
        // Tensor type, we give up extracting the single TensorType inside the
        // subtype.
        function.setArgAttr(index, kGlobalShapeDialectAttr,
                            ConvertTypeToTensorShapeAttr(subtype[0]));
      }
    } else {
      function.setArgAttr(index, kGlobalShapeDialectAttr,
                          ConvertTypeToTensorShapeAttr(argument_type));
    }
  }

  for (const auto& retval_type_and_index :
       llvm::enumerate(function.getFunctionType().getResults())) {
    const int index = retval_type_and_index.index();
    const auto& retval_type = retval_type_and_index.value();
    function.setResultAttr(index, kGlobalShapeDialectAttr,
                           ConvertTypeToTensorShapeAttr(retval_type));
  }
}

// Sets `_global_shape` attribute of an `op` with array of ShapeAttr of
// `outputs.
void AnnotateOperationGlobalShape(mlir::Operation* op,
                                  mlir::OpBuilder* builder) {
  llvm::SmallVector<mlir::Attribute, 4> op_global_shape;
  op_global_shape.reserve(op->getNumResults());

  for (const auto& result_type : op->getResultTypes())
    op_global_shape.emplace_back(ConvertTypeToTensorShapeAttr(result_type));

  op->setAttr(kGlobalShape, builder->getArrayAttr(op_global_shape));
}

// Pass that annotates function argument/return values and all operation with
// `_global_shape` attribute. This will be used during SPMD expansion to
// preserve original global shape of operations in graph after shape has been
// modified to local shape.
struct DTensorAnnotateGlobalShape
    : public impl::DTensorAnnotateGlobalShapeBase<DTensorAnnotateGlobalShape> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);

    auto module = getOperation();
    module.walk([&](mlir::func::FuncOp function) {
      if (function.empty()) return;

      auto* terminator = function.getBody().front().getTerminator();
      AnnotateFunctionArgRetvalGlobalShapes(function, &builder);
      function.getBody().walk([&](mlir::Operation* op) {
        if (op == terminator) return;

        AnnotateOperationGlobalShape(op, &builder);
      });
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorAnnotateGlobalShape() {
  return std::make_unique<DTensorAnnotateGlobalShape>();
}

}  // namespace dtensor
}  // namespace tensorflow
