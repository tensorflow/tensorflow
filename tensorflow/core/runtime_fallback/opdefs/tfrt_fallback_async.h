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
#ifndef TENSORFLOW_CORE_RUNTIME_FALLBACK_OPDEFS_TFRT_FALLBACK_ASYNC_H_
#define TENSORFLOW_CORE_RUNTIME_FALLBACK_OPDEFS_TFRT_FALLBACK_ASYNC_H_

#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "tfrt/compiler/opdefs/tfrt_op_interfaces.h"  // from @tf_runtime
#include "tfrt/compiler/opdefs/tfrt_traits.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/traits.h"  // from @tf_runtime

using namespace mlir;  // NOLINT

namespace tfrt {
namespace fallback_async {

// Dialect for fallback async operations.
class FallbackAsyncDialect : public Dialect {
 public:
  explicit FallbackAsyncDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "tfrt_fallback_async"; }
};

}  // namespace fallback_async
}  // namespace tfrt

#define GET_OP_CLASSES
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback_async.h.inc"

#endif  // TENSORFLOW_CORE_RUNTIME_FALLBACK_OPDEFS_TFRT_FALLBACK_ASYNC_H_
