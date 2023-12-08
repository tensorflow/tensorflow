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

// Special resource type to track TPU Embedding specific ops, which must execute
// but do not have side effects with one another or with resource variable ops.
struct TPUEmbedding : ::mlir::SideEffects::Resource::Base<TPUEmbedding> {
  StringRef getName() final { return "TPUEmbedding"; }
};

// Resource corresponding to GeneratorOp.
struct GeneratorOp : public ::mlir::SideEffects::Resource::Base<GeneratorOp> {
  StringRef getName() final { return "Generator"; }
};

struct Send : public ::mlir::SideEffects::Resource::Base<Send> {
  StringRef getName() final { return "Send"; }
};

struct Recv : public ::mlir::SideEffects::Resource::Base<Recv> {
  StringRef getName() final { return "Recv"; }
};

struct XlaHostCompute
    : public ::mlir::SideEffects::Resource::Base<XlaHostCompute> {
  StringRef getName() final { return "XlaHostCompute"; }
};

struct RandomGenerator
    : public ::mlir::SideEffects::Resource::Base<RandomGenerator> {
  StringRef getName() final { return "RandomGenerator"; }
};

struct TPUExecute : public ::mlir::SideEffects::Resource::Base<TPUExecute> {
  StringRef getName() final { return "TPUExecute"; }
};

struct MustExecute : public ::mlir::SideEffects::Resource::Base<MustExecute> {
  StringRef getName() final { return "MustExecute"; }
};

struct CollectiveReduceOrdering
    : public ::mlir::SideEffects::Resource::Base<CollectiveReduceOrdering> {
  StringRef getName() final { return "CollectiveReduceOrdering"; }
};

struct NcclAllReduceOrdering
    : public ::mlir::SideEffects::Resource::Base<NcclAllReduceOrdering> {
  StringRef getName() final { return "NcclAllReduceOrdering"; }
};

struct GlobalIterId : public ::mlir::SideEffects::Resource::Base<GlobalIterId> {
  StringRef getName() final { return "GlobalIterId"; }
};

struct XlaLaunch : public ::mlir::SideEffects::Resource::Base<XlaLaunch> {
  StringRef getName() final { return "XlaLaunch"; }
};

struct WriteTrainingPredictions
    : public ::mlir::SideEffects::Resource::Base<WriteTrainingPredictions> {
  StringRef getName() final { return "WriteTrainingPredictions"; }
};

struct _XlaRun : public ::mlir::SideEffects::Resource::Base<_XlaRun> {
  StringRef getName() final { return "_XlaRun"; }
};

struct CheckNumerics
    : public ::mlir::SideEffects::Resource::Base<CheckNumerics> {
  StringRef getName() final { return "CheckNumerics"; }
};

// Returns true iff resource type with given ID is only self-dependent, i.e.,
// there are no dependencies to other resource types (including unknown resource
// type).
inline bool IsOnlySelfDependent(TypeID resource_type_id) {
  return resource_type_id == ResourceEffects::Send::getResourceID() ||
         resource_type_id == ResourceEffects::Recv::getResourceID();
}

}  // namespace ResourceEffects
}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_SIDE_EFFECTS_H_
