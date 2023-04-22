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

#include "tensorflow/compiler/mlir/tfjs/ir/tfjs_ops.h"

// Generated dialect defs.
#include "tensorflow/compiler/mlir/tfjs/ir/tfjs_dialect.cc.inc"

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfjs/ir/tfjs_ops.cc.inc"

namespace mlir {
namespace tfjs {

//===----------------------------------------------------------------------===//
// TFJSDialect
//===----------------------------------------------------------------------===//

void TFJSDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tfjs/ir/tfjs_ops.cc.inc"
      >();
}
}  // namespace tfjs
}  // namespace mlir
