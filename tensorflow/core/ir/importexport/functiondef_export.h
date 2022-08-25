/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_IR_IMPORTEXPORT_FUNCTIONDEF_EXPORT_H_
#define TENSORFLOW_CORE_IR_IMPORTEXPORT_FUNCTIONDEF_EXPORT_H_

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "tensorflow/compiler/xla/stream_executor/lib/statusor.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/ir/ops.h"

namespace mlir {
namespace tfg {

// Export a generic GraphFuncOp into a FunctionDef. This is intended to be a
// straight serialization, an error is returned in case of failure.
tensorflow::StatusOr<tensorflow::FunctionDef>
ConvertGenericFunctionToFunctionDef(GraphFuncOp func);

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_IMPORTEXPORT_FUNCTIONDEF_EXPORT_H_
