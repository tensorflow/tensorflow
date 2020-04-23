/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_ATTRIBUTE_IMPORTER_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_ATTRIBUTE_IMPORTER_H_

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Converts an XLA PrecisionConfig to the corresponding MLIR attribute.
mlir::ArrayAttr ConvertPrecisionConfig(const PrecisionConfig* config,
                                       mlir::Builder* builder);

// Converts the gather dimensions to attributes.
mlir::xla_hlo::GatherDimensionNumbers ConvertGatherDimensionNumbers(
    const xla::GatherDimensionNumbers& dnums, mlir::Builder* builder);

// Converts the scatter dimensions to attributes.
mlir::xla_hlo::ScatterDimensionNumbers ConvertScatterDimensionNumbers(
    const xla::ScatterDimensionNumbers& dnums, mlir::Builder* builder);

// Converts the dot dimensions to attributes.
mlir::xla_hlo::DotDimensionNumbers ConvertDotDimensionNumbers(
    const DotDimensionNumbers& dnums, mlir::Builder* builder);

// Converts the conv dimensions to attributes.
mlir::xla_hlo::ConvDimensionNumbers ConvertConvDimensionNumbers(
    const xla::ConvolutionDimensionNumbers& dnums, mlir::Builder* builder);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_ATTRIBUTE_IMPORTER_H_
