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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_LEGALIZE_TF_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_LEGALIZE_TF_H_

#include <memory>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/variant.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/api/v2/device_type.pb.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/client/compile_only_client.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/tpu/kernels/tpu_compile.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {

// Legalizes the given mlir::Module into XLA HLO. If successful, returns the
// compiled XLA HLO. V1 of the tf2xla uses MLIR whereas V0 does not use MLIR.
//
// Inputs:
//  computation - The MLIR module op. It currently takes in
//  tpu::FunctionToHloArgs but this is deprecated. arg_shapes - The shapes of
//  the arguments in module_op. device_type - The device type to compile for.
//  use_tuple_args - Pack the incoming arg shapes into a single tuple.
//  custom_legalization_passes - Extra passes to lower from TF -> MHLO.
//  arg_shapes  - The shapes of the args.
//  arg_core_mapping - Which args go on which cores.
//  per_core_arg_shapes - For each core, the shapes for each argument.
//  client - The Xla Compilation client.
absl::StatusOr<tensorflow::XlaCompilationResult> LegalizeMlirToHlo(
    const std::variant<tpu::MlirToHloArgs, tpu::FunctionToHloArgs>& computation,
    const tpu::TPUCompileMetadataProto& metadata, bool use_tuple_args,
    llvm::StringRef device_type,
    std::vector<std::unique_ptr<mlir::Pass>>& custom_legalization_passes,
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    const std::vector<tensorflow::TensorShape>& arg_shapes,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    xla::CompileOnlyClient* client);

};  // namespace v2
};  // namespace tf2xla
};  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_LEGALIZE_TF_H_
