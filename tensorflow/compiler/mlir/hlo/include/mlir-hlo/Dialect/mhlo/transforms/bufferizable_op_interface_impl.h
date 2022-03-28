/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_DIALECT_MHLO_TRANSFORMS_BUFFERIZABLE_OP_INTERFACE_IMPL_H
#define MLIR_HLO_DIALECT_MHLO_TRANSFORMS_BUFFERIZABLE_OP_INTERFACE_IMPL_H

#include <functional>
#include <memory>

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

namespace mlir {
namespace mhlo {

/// mhlo dialect analysis state. mhlo-specific bufferization options are
/// stored in this state.
struct MhloBufferizationState : public bufferization::DialectAnalysisState {
  using EnforceIdentityMapFn = std::function<bool(Operation *)>;

  /// If this function returns true for an op, copies will be inserted when
  /// the lowering would otherwise lead to a memref with a non-identity map.
  EnforceIdentityMapFn enforce_identity_map_fn = [](Operation *) {
    return true;
  };
};

/// Register the external models for bufferizing mhlo ops.
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

}  // namespace mhlo
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_MHLO_TRANSFORMS_BUFFERIZABLE_OP_INTERFACE_IMPL_H
