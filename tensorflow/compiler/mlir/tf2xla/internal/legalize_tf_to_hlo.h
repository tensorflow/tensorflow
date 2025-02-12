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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_LEGALIZE_TF_TO_HLO_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_LEGALIZE_TF_TO_HLO_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/client/compile_only_client.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

// Legalize the given MLIR module to XLA HLO using a combination of the MLIR
// Bridge and XlaBuilder
absl::StatusOr<XlaCompilationResult> LegalizeTfToHlo(
    const tpu::MlirToHloArgs& computation,
    const tpu::TPUCompileMetadataProto& metadata, bool use_tuple_args,
    llvm::StringRef device_type,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    const std::vector<tensorflow::TensorShape>& arg_shapes,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    std::vector<std::unique_ptr<mlir::Pass>>& custom_legalization_passes,
    xla::CompileOnlyClient* client, XlaCompilationResult* compilation_result);

};  // namespace internal
};  // namespace tf2xla
};  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_LEGALIZE_TF_TO_HLO_H_
