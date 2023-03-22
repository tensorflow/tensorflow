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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_LEGALIZE_TF_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_LEGALIZE_TF_H_

#include <vector>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/api/v1/device_type.pb.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace tf2xla {
namespace v1 {

// Legalizes the given mlir::Module into XLA HLO. If successful, returns the
// compiled XLA HLO.
//
// Inputs:
//  module_op - The MLIR module op.
//  arg_shapes - The shapes of the arguments in module_op.
//  device_type - The device type to compile for.
//  use_tuple_args - Pack the incoming arg shapes into a single tuple.
tsl::StatusOr<tensorflow::XlaCompilationResult> LegalizeMlirToXlaHlo(
    mlir::ModuleOp module_op, const std::vector<TensorShape>& arg_shapes,
    DeviceType device_type, bool use_tuple_args);

};  // namespace v1
};  // namespace tf2xla
};  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V1_LEGALIZE_TF_H_
