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

#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"

#include <string>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir

namespace tensorflow {

OpOrArgNameMapper::~OpOrArgNameMapper() {}

std::string OpOrArgNameMapper::GetUniqueName(llvm::StringRef prefix) {
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

const std::string& OpOrArgNameMapper::GetUniqueName(OpOrArg op_or_arg) {
  auto& name = op_or_arg_to_name_[op_or_arg];
  if (!name.empty()) return name;
  // Update the value in the map with unique name.
  name = GetUniqueName(GetName(op_or_arg));
  return name;
}

int OpOrArgNameMapper::InitOpName(OpOrArg op_or_arg, llvm::StringRef name) {
  op_or_arg_to_name_[op_or_arg] = name;
  return name_to_count_[name]++;
}

bool OpOrArgNameMapper::IsUnique(llvm::StringRef name) {
  return name_to_count_.count(name) == 0;
}

namespace {
// Derives name from location.
llvm::StringRef GetNameFromLoc(mlir::Location loc) {
  if (auto name_loc = loc.dyn_cast<mlir::NameLoc>())
    return name_loc.getName().strref();

  if (auto call_loc = loc.dyn_cast<mlir::CallSiteLoc>()) {
    // Return name if CallSiteLoc's callee has a NameLoc (as should be the case
    // if imported with DebugInfo), else use the fallback naming scheme below.
    if (auto name_loc = call_loc.getCallee().dyn_cast<mlir::NameLoc>())
      return name_loc.getName().strref();
  }

  return llvm::StringRef();
}
}  // anonymous namespace

std::string OpOrArgLocNameMapper::GetName(OpOrArg op_or_arg) {
  if (auto* op = op_or_arg.dyn_cast<mlir::Operation*>()) {
    auto name_from_loc = GetNameFromLoc(op->getLoc());
    if (!name_from_loc.empty()) return name_from_loc;
    // If the location is none of the expected types, then simply use name
    // generated using the op type.
    return op->getName().getStringRef();
  }

  if (auto* arg = op_or_arg.dyn_cast<mlir::BlockArgument*>())
    return GetNameFromLoc(arg->getLoc());

  return "";
}

std::string OpOrArgStripNameMapper::GetName(OpOrArg op_or_arg) {
  return llvm::APInt(32, count_++)
      .toString(/*Radix=*/36,
                /*Signed=*/false);
}

}  // namespace tensorflow
