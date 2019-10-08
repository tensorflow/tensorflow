//===- StringExtras.h - String utilities used by MLIR -----------*- C++ -*-===//
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
// This file contains string utility functions used within MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_STRINGEXTRAS_H
#define MLIR_SUPPORT_STRINGEXTRAS_H

#include "llvm/ADT/StringExtras.h"

#include <cctype>

namespace mlir {
/// Converts a string to snake-case from camel-case by replacing all uppercase
/// letters with '_' followed by the letter in lowercase, except if the
/// uppercase letter is the first character of the string.
inline std::string convertToSnakeCase(llvm::StringRef input) {
  std::string snakeCase;
  snakeCase.reserve(input.size());
  for (auto c : input) {
    if (std::isupper(c)) {
      if (!snakeCase.empty() && snakeCase.back() != '_') {
        snakeCase.push_back('_');
      }
      snakeCase.push_back(llvm::toLower(c));
    } else {
      snakeCase.push_back(c);
    }
  }
  return snakeCase;
}

/// Converts a string from camel-case to snake_case by replacing all occurrences
/// of '_' followed by a lowercase letter with the letter in
/// uppercase. Optionally allow capitalization of the first letter (if it is a
/// lowercase letter)
inline std::string convertToCamelCase(llvm::StringRef input,
                                      bool capitalizeFirst = false) {
  if (input.empty()) {
    return "";
  }
  std::string output;
  output.reserve(input.size());
  size_t pos = 0;
  if (capitalizeFirst && std::islower(input[pos])) {
    output.push_back(llvm::toUpper(input[pos]));
    pos++;
  }
  while (pos < input.size()) {
    auto cur = input[pos];
    if (cur == '_') {
      if (pos && (pos + 1 < input.size())) {
        if (std::islower(input[pos + 1])) {
          output.push_back(llvm::toUpper(input[pos + 1]));
          pos += 2;
          continue;
        }
      }
    }
    output.push_back(cur);
    pos++;
  }
  return output;
}
} // namespace mlir

#endif // MLIR_SUPPORT_STRINGEXTRAS_H
