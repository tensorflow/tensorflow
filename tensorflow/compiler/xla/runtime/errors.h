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

#ifndef XLA_RUNTIME_ERRORS_H_
#define XLA_RUNTIME_ERRORS_H_

#include <string>

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace xla {
namespace runtime {

// TODO(ezhulenev): Replace all uses of llvm errors inside the runtime with ABSL
// types: Error -> Status, Expected -> StatusOr.

namespace internal {

template <typename StreamT>
inline void ToStreamHelper(StreamT& os) {}

template <typename StreamT, typename T, typename... Args>
void ToStreamHelper(StreamT& os, T&& v, Args&&... args) {
  os << std::forward<T>(v);
  ToStreamHelper(os, std::forward<Args>(args)...);
}

template <typename... Args>
std::string StrCat(Args&&... args) {
  std::string str;
  llvm::raw_string_ostream sstr(str);
  internal::ToStreamHelper(sstr, std::forward<Args>(args)...);
  sstr.flush();
  return str;
}

}  // namespace internal

template <typename... Args>
llvm::Error MakeStringError(Args&&... args) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 internal::StrCat(std::forward<Args>(args)...));
}

}  // namespace runtime
}  // namespace xla

#endif  // XLA_RUNTIME_ERRORS_H_
