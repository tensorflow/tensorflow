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

#include "tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_ops.h"

#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace mlir {
namespace tfd {

//===----------------------------------------------------------------------===//
// TfrtDelegate Dialect
//===----------------------------------------------------------------------===//

RuntimeFallbackDialect::RuntimeFallbackDialect(MLIRContext *context)
    : Dialect(/*name=*/"tfd", context, TypeID::get<RuntimeFallbackDialect>()) {
  allowUnknownTypes();

  allowUnknownOperations();

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tfrt/runtime_fallback_ops.cc.inc"
      >();
}

}  // namespace tfd
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfrt/runtime_fallback_ops.cc.inc"
