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

#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_dialect.h"

#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project  // IWYU pragma: keep
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_ops.h"  // IWYU pragma: keep

//===----------------------------------------------------------------------===//
// XLA GPU Dialect
//===----------------------------------------------------------------------===//

#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_dialect.cc.inc"

namespace xla::gpu {

void XlaGpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_ops.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_types.cc.inc"
      >();
}

}  // namespace xla::gpu

#define GET_TYPEDEF_CLASSES
#include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_types.cc.inc"
