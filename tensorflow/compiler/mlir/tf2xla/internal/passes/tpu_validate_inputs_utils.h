/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_PASSES_TPU_VALIDATE_INPUTS_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_PASSES_TPU_VALIDATE_INPUTS_UTILS_H_

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

constexpr char kTpuReplicatedCoreZeroAttr[] = "TPU_REPLICATED_CORE:0";

using mlir::ModuleOp;
using mlir::Operation;
using mlir::StringAttr;
using mlir::TypeID;
using mlir::TF::InfeedDequeueTupleOp;
using mlir::TF::kDeviceAttr;
using mlir::tf_executor::GraphOp;

bool IsPotentialUnsupportedOp(Operation* op);

bool HasV1ControlFlow(GraphOp graph);

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_INTERNAL_PASSES_TPU_VALIDATE_INPUTS_UTILS_H_
