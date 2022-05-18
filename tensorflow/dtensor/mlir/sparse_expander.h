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

#ifndef TENSORFLOW_DTENSOR_MLIR_SPARSE_EXPANDER_H_
#define TENSORFLOW_DTENSOR_MLIR_SPARSE_EXPANDER_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/dtensor/cc/dstatus.h"

namespace tensorflow {
namespace dtensor {

// Base class for handling Sparse expansion of a MLIR TF Operation.
// Note that an op will only go through Sparse Expansion only if it has
// any sparse input tensors.
class SparseExpanderBase {
 public:
  virtual ~SparseExpanderBase() {}

  // Converts `op` to a Sparse expanded form. Sparse expansion logic is
  // a function of op type and op's operand type.
  // Must return the `op` that is expanded as the final return value.
  //
  // An op has a SparseTensor operand if the defining op of that operand
  // is a SparseToDenseOp.
  virtual StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) = 0;
};

// Computes the Sparse expansion for `op`.
Status RunSparseExpansion(mlir::Operation* op, mlir::Operation** output);

// A registry of sparse SPMD expanders. This map is statically stored and
// initialized with all the registered sparse SPMD expanders.
class SparseExpanderRegistry {
 public:
  ~SparseExpanderRegistry() = default;

  // A singleton available at startup.
  static SparseExpanderRegistry* Global();

  // Returns the sparse expansion for the given operation (or nullptr if no
  // expansion has been registered).
  SparseExpanderBase* GetSparseExpansionFnForOp(mlir::Operation* op);

  // Registers a sparse expander for the provided opName.
  InitOnStartupMarker RegisterSparseExpansionFn(
      std::string opName, std::unique_ptr<SparseExpanderBase> prop);

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<SparseExpanderBase>>
      op_to_sparse_expansion_fn_map_;
};

#define REGISTER_SPARSE(name, op, prop, ...)                          \
  static ::tensorflow::InitOnStartupMarker const spmd_##name =        \
      InitOnStartupMarker{}                                           \
      << SparseExpanderRegistry::Global()->RegisterSparseExpansionFn( \
             mlir::op ::getOperationName().str(),                     \
             absl::make_unique<prop>(__VA_ARGS__))

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_SPARSE_EXPANDER_H_
