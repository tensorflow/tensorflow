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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_RESOURCE_VALUE_TYPED_ANALYZER_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_RESOURCE_VALUE_TYPED_ANALYZER_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace mlir {
namespace TF {

class ResourceAnalyzer {
 public:
  explicit ResourceAnalyzer(ModuleOp module, bool skip_session_init = false);

  bool IsPotentiallyWritten(Value resource) const;

 private:
  // Analyze the specified region for resource mutating operations, namely
  // TF::AssignVariableOp, if so, set the resource associated as "potentially
  // written".
  LogicalResult AnalyzeRegion(Region& region);

  // If an op is not one of the handled ones, we assume all resource usages
  // within its purview are mutating in nature.
  void PropagatePotentiallyWrittenWithinUnhandledOp(Operation* op);

  // Given a Region associated with the callee and operands from the
  // corresponding callOp, propagate the potentially written decision to the
  // callOp's operands, if the corresponding region's arguments are potentially
  // written resources.
  void PropagatePotentiallyWrittenUpFromCallee(
      Region& region, Operation::operand_range propagate_to);

  // Marks 'resource' as written.
  void SetPotentiallyWritten(Value resource);

  struct ResourceInfo {
    bool potentially_written = false;
  };
  // Key: Resource Value's
  // Value: Information we know about that Value.
  // Note that these Value's are in general in different functions.
  DenseMap<Value, ResourceInfo> resource_infos_;
  // The set of regions we already discovered.
  DenseSet<Region*> discovered_;
  // Identifiers about mutable variables.
  // All variables are identified by (device, container, shared_name).
  DenseSet<std::tuple<llvm::StringRef, llvm::StringRef, llvm::StringRef>>
      mutable_variables_;
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_RESOURCE_VALUE_TYPED_ANALYZER_H_
