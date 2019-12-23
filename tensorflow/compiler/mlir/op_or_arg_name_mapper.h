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

#ifndef TENSORFLOW_COMPILER_MLIR_op_or_val_NAME_MAPPER_H_
#define TENSORFLOW_COMPILER_MLIR_op_or_val_NAME_MAPPER_H_

#include <string>

#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir

namespace tensorflow {

// PointerUnion for operation and value.
// TODO(jpienaar): Rename the files.
using OpOrVal = llvm::PointerUnion<mlir::Operation*, mlir::Value>;

// Mapper from operation or value to name.
class OpOrArgNameMapper {
 public:
  // Returns unique name for the given prefix.
  llvm::StringRef GetUniqueName(llvm::StringRef prefix);

  // Returns unique name for the operation or value.
  llvm::StringRef GetUniqueName(OpOrVal op_or_val);

  // Returns unique name as a string_view for the operation or value.
  absl::string_view GetUniqueNameView(OpOrVal op_or_val);

  // Initializes operation or value to map to name. Returns number of
  // operations or value already named 'name' which should be 0 else
  // GetUniqueName could return the same names for different operations or
  // values.
  // Note: Its up to the caller to decide the behavior when assigning two
  // operations or values to the same name.
  int InitOpName(OpOrVal op_or_val, llvm::StringRef name);

  virtual ~OpOrArgNameMapper();

 protected:
  // Returns true if the name is unique. A derived class can override it if the
  // class maintains uniqueness in a different scope.
  virtual bool IsUnique(llvm::StringRef name);

  // Returns a constant view of the underlying map.
  const llvm::DenseMap<OpOrVal, absl::string_view>& GetMap() const {
    return op_or_val_to_name_;
  }

 private:
  // Returns name from the location of the operation or value.
  virtual std::string GetName(OpOrVal op_or_val) = 0;

  // Maps string name to count. This map is used to help keep track of unique
  // names for operations or values.
  llvm::StringMap<int64_t> name_to_count_;
  // Maps operation or values to name. Value in map is a view of the string
  // name in `name_to_count_`. Names in `name_to_count_` are never removed.
  llvm::DenseMap<OpOrVal, absl::string_view> op_or_val_to_name_;
};

// OpOrArgNameMapper that returns, for operations or values not initialized
// to a specific name, a name based on the location of the operation or
// value.
class OpOrArgLocNameMapper : public OpOrArgNameMapper {
 private:
  std::string GetName(OpOrVal op_or_val) override;
};

// OpOrArgNameMapper that returns, for operations or values not initialized
// to a specific name, a short name.
class OpOrArgStripNameMapper : public OpOrArgNameMapper {
 private:
  std::string GetName(OpOrVal op_or_val) override;

  // Number of ops mapped.
  int count_ = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_op_or_val_NAME_MAPPER_H_
