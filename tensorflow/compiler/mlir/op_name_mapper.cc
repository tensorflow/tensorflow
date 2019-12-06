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

#include "tensorflow/compiler/mlir/op_name_mapper.h"

#include "llvm/ADT/APInt.h"

namespace tensorflow {

using llvm::StringRef;
using mlir::Operation;

OpNameMapper::~OpNameMapper() {}

std::string OpNameMapper::GetUniqueName(llvm::StringRef prefix) {
  std::string name = prefix;
  if (IsUnique(name)) {
    ++name_to_count_[name];
    return name;
  }

  auto& val = name_to_count_[name];
  llvm::SmallString<64> probe_name(prefix);
  while (true) {
    probe_name.resize(prefix.size());
    // TODO(jpienaar): Subtract one so that the initial suffix is 0 instead
    // of 1.
    // TODO(jpienaar): Switch to radix 36 and update tests.
    llvm::APInt(32, val++).toString(probe_name, /*Radix=*/10,
                                    /*Signed=*/false);
    if (IsUnique(probe_name)) {
      name = llvm::StringRef(probe_name);
      ++name_to_count_[name];
      break;
    }
  }
  return name;
}

const std::string& OpNameMapper::GetUniqueName(Operation* op) {
  auto& name = op_to_name_[op];
  if (!name.empty()) return name;
  // Update the value in the map with unique name.
  name = GetUniqueName(GetName(op));
  return name;
}

int OpNameMapper::InitOpName(mlir::Operation* op, llvm::StringRef name) {
  op_to_name_[op] = name;
  return name_to_count_[name]++;
}

bool OpNameMapper::IsUnique(llvm::StringRef name) {
  return name_to_count_.count(name) == 0;
}

std::string OpLocNameMapper::GetName(Operation* op) {
  if (auto name_loc = op->getLoc().dyn_cast<mlir::NameLoc>())
    return name_loc.getName().str();

  if (auto call_loc = op->getLoc().dyn_cast<mlir::CallSiteLoc>()) {
    // Return name if CallSiteLoc's callee has a NameLoc (as should be the case
    // if imported with DebugInfo), else use the fallback naming scheme below.
    if (auto name_loc = call_loc.getCallee().dyn_cast<mlir::NameLoc>())
      return name_loc.getName().str();
  }

  // If the location is none of the expected types, then simply use name
  // generated using the op type.
  return op->getName().getStringRef();
}

std::string OpStripNameMapper::GetName(Operation* op) {
  return llvm::APInt(32, count_++)
      .toString(/*Radix=*/36,
                /*Signed=*/false);
}

}  // namespace tensorflow
