/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_targets.h"

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace mhlo {

ConversionTarget GetDefaultLegalConversionTargets(MLIRContext& mlir_context,
                                                  bool legalize_chlo) {
  ConversionTarget target(mlir_context);

  if (legalize_chlo) {
    target.addIllegalDialect<chlo::ChloDialect>();
  } else {
    target.addLegalDialect<chlo::ChloDialect>();
  }
  target.addLegalDialect<MhloDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalDialect<tensor::TensorDialect>();
  target.addLegalDialect<shape::ShapeDialect>();
  target.addLegalOp<func::CallOp>();

  // These ops are legalized in LegalizeTFCommunication after this and that pass
  // only operates on MHLO control flow ops.
  target.addLegalOp<TF::_XlaHostComputeMlirOp, TF::XlaSendToHostOp,
                    TF::XlaRecvFromHostOp>();

  return target;
}

}  // namespace mhlo
}  // namespace mlir
