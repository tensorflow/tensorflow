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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_LEGALIZE_TF_MLIR_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_LEGALIZE_TF_MLIR_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

// Runs all the MLIR Bridge passes on the given MLIR module.
// If compile_to_xla_hlo is true then those passes include all the Legalization
// to XLA HLO which is returned in the compilation_result.
absl::Status CompileFromMlirToXlaHlo(
    bool lower_to_xla_hlo, mlir::ModuleOp mlir_module_op,
    const tpu::TPUCompileMetadataProto& metadata, llvm::StringRef device_type,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns& shape_determination_fns,
    bool use_tuple_args, XlaCompiler::CompilationResult* compilation_result,
    std::vector<std::unique_ptr<mlir::Pass>>& custom_legalization_passes,
    const std::vector<TensorShape>& arg_shapes,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes);

};  // namespace internal
};  // namespace tf2xla
};  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_LEGALIZE_TF_MLIR_H_
