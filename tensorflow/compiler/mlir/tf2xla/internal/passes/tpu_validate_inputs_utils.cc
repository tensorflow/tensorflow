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

#include "tensorflow/compiler/mlir/tf2xla/internal/passes/tpu_validate_inputs_utils.h"

#include <string>

#include "absl/strings/match.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Support/TypeID.h"  // from @llvm-project

namespace tensorflow {
namespace tf2xla {
namespace internal {

bool IsPotentialUnsupportedOp(Operation* op) {
  static auto* ops = [] {
    llvm::SmallDenseSet<mlir::TypeID, 32>* ops_set =
        new llvm::SmallDenseSet<mlir::TypeID, 32>{
            TypeID::get<InfeedDequeueTupleOp>(),
        };
    return ops_set;
  }();
  auto abstractOp = op->getRegisteredInfo();
  if (!abstractOp) return false;

  bool is_in_ops = ops->count(abstractOp->getTypeID()) != 0;
  if (!is_in_ops) return false;

  std::string device = "";
  if (!op->hasAttr(kDeviceAttr)) return false;
  device = op->getAttrOfType<StringAttr>(kDeviceAttr).str();
  if (!absl::StrContains(device, kTpuReplicatedCoreZeroAttr)) return false;
  op->emitWarning("TPU_REPLICARTED_CORE:0 device is not supported for op = ")
      << op->getName() << " in TF2XLA MLIR Bridge";

  return true;
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
