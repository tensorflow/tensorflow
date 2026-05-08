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
  StringRef getName() const final { return "Variable"; }
};

struct Stack : ::mlir::SideEffects::Resource::Base<Stack> {
  StringRef getName() const final { return "Stack"; }
};

struct TensorArray : ::mlir::SideEffects::Resource::Base<TensorArray> {
  StringRef getName() const final { return "TensorArray"; }
};

struct Summary : ::mlir::SideEffects::Resource::Base<Summary> {
  StringRef getName() const final { return "Summary"; }
};

struct LookupTable : ::mlir::SideEffects::Resource::Base<LookupTable> {
  StringRef getName() const final { return "LookupTable"; }
};

struct DatasetSeedGenerator
    : ::mlir::SideEffects::Resource::Base<DatasetSeedGenerator> {
  StringRef getName() const final { return "DatasetSeedGenerator"; }
};

struct DatasetMemoryCache
    : ::mlir::SideEffects::Resource::Base<DatasetMemoryCache> {
  StringRef getName() const final { return "DatasetMemoryCache"; }
};

struct DatasetIterator : ::mlir::SideEffects::Resource::Base<DatasetIterator> {
  StringRef getName() const final { return "DatasetIterator"; }
};

// Special resource type to track TPU Embedding specific ops, which must execute
// but do not have side effects with one another or with resource variable ops.
struct TPUEmbedding : ::mlir::SideEffects::Resource::Base<TPUEmbedding> {
  StringRef getName() const final { return "TPUEmbedding"; }
};

// Resource corresponding to GeneratorOp.
struct GeneratorOp : public ::mlir::SideEffects::Resource::Base<GeneratorOp> {
  StringRef getName() const final { return "Generator"; }
};

struct Send : public ::mlir::SideEffects::Resource::Base<Send> {
  StringRef getName() const final { return "Send"; }
};

struct Recv : public ::mlir::SideEffects::Resource::Base<Recv> {
  StringRef getName() const final { return "Recv"; }
};

struct XlaHostCompute
    : public ::mlir::SideEffects::Resource::Base<XlaHostCompute> {
  StringRef getName() const final { return "XlaHostCompute"; }
};

struct RandomGenerator
    : public ::mlir::SideEffects::Resource::Base<RandomGenerator> {
  StringRef getName() const final { return "RandomGenerator"; }
};

struct TPUExecute : public ::mlir::SideEffects::Resource::Base<TPUExecute> {
  StringRef getName() const final { return "TPUExecute"; }
};

struct MustExecute : public ::mlir::SideEffects::Resource::Base<MustExecute> {
  StringRef getName() const final { return "MustExecute"; }
};

struct CollectiveReduceOrdering
    : public ::mlir::SideEffects::Resource::Base<CollectiveReduceOrdering> {
  StringRef getName() const final { return "CollectiveReduceOrdering"; }
};

struct NcclAllReduceOrdering
    : public ::mlir::SideEffects::Resource::Base<NcclAllReduceOrdering> {
  StringRef getName() const final { return "NcclAllReduceOrdering"; }
};

struct GlobalIterId : public ::mlir::SideEffects::Resource::Base<GlobalIterId> {
  StringRef getName() const final { return "GlobalIterId"; }
};

struct XlaLaunch : public ::mlir::SideEffects::Resource::Base<XlaLaunch> {
  StringRef getName() const final { return "XlaLaunch"; }
};

struct WriteTrainingPredictions
    : public ::mlir::SideEffects::Resource::Base<WriteTrainingPredictions> {
  StringRef getName() const final { return "WriteTrainingPredictions"; }
};

struct RecordEventMetricForTensor
    : public ::mlir::SideEffects::Resource::Base<RecordEventMetricForTensor> {
  StringRef getName() const final { return "RecordEventMetricForTensor"; }
};

struct _XlaRun : public ::mlir::SideEffects::Resource::Base<_XlaRun> {
  StringRef getName() const final { return "_XlaRun"; }
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
