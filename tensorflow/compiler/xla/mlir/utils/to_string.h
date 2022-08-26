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

#ifndef XLA_MLIR_UTILS_TO_STRING_H_
#define XLA_MLIR_UTILS_TO_STRING_H_

#include <string>
#include <utility>

#include "llvm/Support/raw_ostream.h"

namespace xla {

// Converts to string values that can be printed to the llvm::raw_ostream, e.g.
// it can convert to string MLIR types, attributes and values.
template <
    typename T,
    std::enable_if_t<std::is_same_v<llvm::raw_ostream&&,
                                    decltype(std::declval<llvm::raw_ostream>()
                                             << std::declval<T>())>,
                     void>* = nullptr>
std::string ToString(T value) {
  std::string str;
  llvm::raw_string_ostream(str) << value;
  return str;
}

}  // namespace xla

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_XLA_MLIR_UTILS_TO_STRING_H_
