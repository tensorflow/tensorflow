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

// This is the side effect definition file for TensorFlow.
#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_SIDE_EFFECTS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_SIDE_EFFECTS_H_

#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project

namespace mlir {
namespace TF {
namespace ResourceEffects {

struct Variable : ::mlir::SideEffects::Resource::Base<Variable> {
  StringRef getName() final { return "Variable"; }
};

struct Stack : ::mlir::SideEffects::Resource::Base<Stack> {
  StringRef getName() final { return "Stack"; }
};

struct TensorArray : ::mlir::SideEffects::Resource::Base<TensorArray> {
  StringRef getName() final { return "TensorArray"; }
};

struct Summary : ::mlir::SideEffects::Resource::Base<Summary> {
  StringRef getName() final { return "Summary"; }
};

struct LookupTable : ::mlir::SideEffects::Resource::Base<LookupTable> {
  StringRef getName() final { return "LookupTable"; }
};

struct DatasetSeedGenerator
    : ::mlir::SideEffects::Resource::Base<DatasetSeedGenerator> {
  StringRef getName() final { return "DatasetSeedGenerator"; }
};

struct DatasetMemoryCache
    : ::mlir::SideEffects::Resource::Base<DatasetMemoryCache> {
  StringRef getName() final { return "DatasetMemoryCache"; }
};

struct DatasetIterator : ::mlir::SideEffects::Resource::Base<DatasetIterator> {
  StringRef getName() final { return "DatasetIterator"; }
};

}  // namespace ResourceEffects
}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_SIDE_EFFECTS_H_
