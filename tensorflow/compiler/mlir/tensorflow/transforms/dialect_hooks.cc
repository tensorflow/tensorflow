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

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Dialect.h"  // TF:local_config_mlir
#include "mlir/IR/DialectHooks.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "mlir/IR/Types.h"  // TF:local_config_mlir
#include "mlir/Support/LogicalResult.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/constant_fold.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/core/framework/logging.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace mlir {
namespace {

// Since this method is passed to MLIR as decode hook it has to conform
// to LLVM style used by MLIR.
bool DecodeOpaqueTensorHook(const OpaqueElementsAttr input,
                            ElementsAttr& output) {  // NOLINT
  Builder builder(input.getType().getContext());
  auto decoded_attr_or = tensorflow::DecodeOpaqueTensor(input, builder);
  if (!decoded_attr_or.ok()) {
    VLOG(2) << decoded_attr_or.status().error_message();
    return true;
  }

  output = decoded_attr_or.ValueOrDie();
  return false;
}

// Hooks for the TensorFlow dialect.
class TensorFlowHooks : public DialectHooks {
 public:
  DialectConstantFoldHook getConstantFoldHook() {
    return TF::ConstantFoldFallbackHook;
  }
  DialectConstantDecodeHook getDecodeHook() { return DecodeOpaqueTensorHook; }
};

}  // anonymous namespace

// Static initialization for TensorFlow dialect hooks registration.
static DialectHooksRegistration<TensorFlowHooks> tf_hooks_registration("tf");

}  // namespace mlir
