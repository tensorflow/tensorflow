/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/quantization/lite/tfl_to_std.h"

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"

namespace mlir {
namespace TFL {

void ConvertTFLQuantOpsToMlirQuantOps(FuncOp func) {
  OpBuilder b(func);
  func.walk([&](Operation* op) {
    b.setInsertionPoint(op);
    if (auto dq = llvm::dyn_cast<DequantizeOp>(op)) {
      auto dcast = b.create<quant::DequantizeCastOp>(
          dq.getLoc(), dq.output().getType(), dq.input());
      dq.output().replaceAllUsesWith(dcast);
      dq.erase();
    } else if (auto q = llvm::dyn_cast<QuantizeOp>(op)) {
      auto qcast = b.create<quant::QuantizeCastOp>(
          q.getLoc(), q.output().getType(), q.input());
      q.output().replaceAllUsesWith(qcast);
      q.erase();
    } else if (auto q = llvm::dyn_cast<ConstOp>(op)) {
      auto value = q.value();
      auto type = q.getResult().getType();
      if (arith::ConstantOp::isBuildableWith(value, type)) {
        auto c = b.create<arith::ConstantOp>(q.getLoc(), q.value());
        q.output().replaceAllUsesWith(c);
        q.erase();
      } else if (ConstantOp::isBuildableWith(value, type)) {
        auto c = b.create<ConstantOp>(q.getLoc(), q.value());
        q.output().replaceAllUsesWith(c);
        q.erase();
      }
    }
  });
}

void ConvertMlirQuantOpsToTFLQuantOps(FuncOp func) {
  OpBuilder b(func);
  func.walk([&](Operation* op) {
    b.setInsertionPoint(op);
    if (auto dq = llvm::dyn_cast<quant::DequantizeCastOp>(op)) {
      auto dcast = b.create<DequantizeOp>(dq.getLoc(), dq.getResult().getType(),
                                          dq.arg());
      dq.getResult().replaceAllUsesWith(dcast);
      if (auto extra_attr = op->getAttr(mlir::quant::kVolatileOpAttrName)) {
        dcast->setAttr(mlir::quant::kVolatileOpAttrName, extra_attr);
      }
      dq.erase();
    } else if (auto q = llvm::dyn_cast<quant::QuantizeCastOp>(op)) {
      auto out_type = q.getResult().getType();
      auto qcast = b.create<QuantizeOp>(q.getLoc(), out_type, q.arg(),
                                        TypeAttr::get(out_type));
      q.getResult().replaceAllUsesWith(qcast);
      if (auto extra_attr = op->getAttr(mlir::quant::kVolatileOpAttrName)) {
        qcast->setAttr(mlir::quant::kVolatileOpAttrName, extra_attr);
      }
      q.erase();
    }
  });
}

}  // namespace TFL
}  // namespace mlir
