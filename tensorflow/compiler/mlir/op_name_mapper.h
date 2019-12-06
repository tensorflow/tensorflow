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

#ifndef TENSORFLOW_COMPILER_MLIR_OP_NAME_MAPPER_H_
#define TENSORFLOW_COMPILER_MLIR_OP_NAME_MAPPER_H_

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"  // TF:local_config_mlir

namespace tensorflow {

// Mapper from operation to name.
class OpNameMapper {
 public:
  // Returns unique name for the operation.
  const std::string& GetUniqueName(mlir::Operation* op);

  // Returns unique name for the given prefix.
  std::string GetUniqueName(llvm::StringRef prefix);

  // Initializes operation to map to name. Returns number of operations already
  // named 'name' which should be 0 else GetUniqueName could return the same
  // names for different ops.
  // Note: its up to the caller to decide the behavior when assigning two ops
  // to the same name.
  int InitOpName(mlir::Operation* op, llvm::StringRef name);

  virtual ~OpNameMapper();

 protected:
  // Returns true if the name is unique. A derived class can override it if the
  // class maintains uniqueness in a different scope.
  virtual bool IsUnique(llvm::StringRef name);

 private:
  // Returns name from the location of the operation.
  virtual std::string GetName(mlir::Operation* op) = 0;

  // Maps from op to name.
  llvm::StringMap<int64_t> name_to_count_;
  absl::flat_hash_map<mlir::Operation*, std::string> op_to_name_;
};

// OpNameMapper that returns, for ops not initialized to a specific name, a name
// based on the location of the operation.
class OpLocNameMapper : public OpNameMapper {
 private:
  std::string GetName(mlir::Operation* op) override;
};

// OpNameMapper that returns, for ops not initialized to a specific name, a
// short name.
class OpStripNameMapper : public OpNameMapper {
 private:
  std::string GetName(mlir::Operation* op) override;

  // Number of ops mapped.
  int count_ = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_OP_NAME_MAPPER_H_
