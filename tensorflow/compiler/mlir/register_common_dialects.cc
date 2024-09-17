/* Copyright 2023 Google Inc. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/register_common_dialects.h"

#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/InitAllExtensions.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mlprogram_util.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "xla/mlir/framework/ir/xla_framework.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"

namespace mlir {

void RegisterCommonToolingDialects(mlir::DialectRegistry& registry) {
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::stablehlo::registerAllDialects(registry);

  registry.insert<mlir::TFL::TensorFlowLiteDialect>();
  registry.insert<mlir::kernel_gen::tf_framework::TFFrameworkDialect>();
  registry.insert<mlir::quant::QuantizationDialect>();
  registry.insert<mlir::quantfork::QuantizationForkDialect>();
  registry.insert<mlir::shape::ShapeDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::tosa::TosaDialect>();
  registry.insert<mlir::xla_framework::XLAFrameworkDialect,
                  mlir::TF::TensorFlowDialect, mlir::tf_type::TFTypeDialect>();
}

};  // namespace mlir
