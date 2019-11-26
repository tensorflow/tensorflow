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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_MLIR_HLO_TO_HLO_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_MLIR_HLO_TO_HLO_H_

#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace mlir {

// Converts a MLIR module in HLO dialect into a HloModuleProto. If
// use_tuple_args is set, then the entry computations's arguments are converted
// to a tuple and passed as a single parameter.
// Similarly, if return tuple is true, then the entry function's return values
// are converted to a tuple even when there is only a single return value.
// Multiple return values are always converted to a tuple and returned as a
// single value.
Status ConvertMlirHloToHlo(mlir::ModuleOp module, xla::HloProto* hlo_proto,
                           bool use_tuple_args, bool return_tuple);

// Creates XlaOp equivalent of a given MLIR operation using the operand info
// from `value_lowering` map.
llvm::Optional<xla::XlaOp> CreateXlaOperator(
    mlir::Operation* op,
    llvm::DenseMap<mlir::Value*, xla::XlaOp>* value_lowering);

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_MLIR_HLO_TO_HLO_H_
