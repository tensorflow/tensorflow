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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_ATTRIBUTE_EXPORTER_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_ATTRIBUTE_EXPORTER_H_

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/stream_executor/dnn.h"

namespace xla {

// Converts the conv dimensions attribute to XLA HLO.
ConvolutionDimensionNumbers ConvertConvDimensionNumbers(
    mlir::mhlo::ConvDimensionNumbersAttr input);

StatusOr<stream_executor::dnn::ActivationMode> ConvertConvActivationMode(
    llvm::StringRef input);

StatusOr<std::vector<ReplicaGroup>> ConvertReplicaGroups(
    mlir::DenseIntElementsAttr input);

// Convert a (N, 2) dense attribute to a list of tuples. This is the way padding
// and source-target pairs are defined in HLO.
StatusOr<std::vector<std::pair<int64_t, int64_t>>> ConvertNx2Attribute(
    llvm::Optional<mlir::DenseIntElementsAttr> optional_attr);

StatusOr<FftType> ConvertFftType(llvm::StringRef type_string);
StatusOr<TriangularSolveOptions::Transpose> ConvertTranspose(
    llvm::StringRef transpose_string);

StatusOr<xla::CustomCallApiVersion> ConvertCustomCallApiVersion(
    mlir::mhlo::CustomCallApiVersion api_version);

}  // namespace xla
#endif  // TENSORFLOW_COMPILER_MLIR_XLA_ATTRIBUTE_EXPORTER_H_
