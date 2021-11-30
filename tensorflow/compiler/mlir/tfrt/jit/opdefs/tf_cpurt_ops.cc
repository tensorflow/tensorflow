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

#include "tensorflow/compiler/mlir/tfrt/jit/opdefs/tf_cpurt_ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace mlir {
namespace tf_cpurt {

//===----------------------------------------------------------------------===//
// CpuRuntimeDialect Dialect
//===----------------------------------------------------------------------===//

CpuRuntimeDialect::CpuRuntimeDialect(mlir::MLIRContext *context)
    : Dialect(/*name*/ "tf_cpurt", context,
              mlir::TypeID::get<CpuRuntimeDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tfrt/tf_cpurt_ops.cc.inc"
      >();
}

}  // namespace tf_cpurt
}  // end namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfrt/tf_cpurt_ops.cc.inc"
