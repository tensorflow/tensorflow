//===- DebugStringHelper.h - helpers to generate debug strings --*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Convenience functions to make it easier to get a string representation for
// ops that have a print method. For use in debugging output and errors
// returned.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DEBUGSTRINGHELPER_H_
#define MLIR_DEBUGSTRINGHELPER_H_

#include <string>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

// Simple helper function that returns a string as printed from a op.
template <typename T> static std::string debugString(T &op) {
  std::string instr_str;
  llvm::raw_string_ostream os(instr_str);
  op.print(os);
  return os.str();
}

} // namespace mlir

inline std::ostream &operator<<(std::ostream &out, const llvm::Twine &twine) {
  llvm::raw_os_ostream rout(out);
  rout << twine;
  return out;
}

#endif // MLIR_DEBUGSTRINGHELPER_H_
