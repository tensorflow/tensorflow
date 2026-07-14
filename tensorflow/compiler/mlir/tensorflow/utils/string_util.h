/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_STRING_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_STRING_UTIL_H_

#include <ostream>
#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project

// Utility functions for dumping operations/attributes as strings and ostream
// bindings.

namespace tensorflow {
std::string OpAsString(mlir::Operation& op);
std::string AttrAsString(mlir::Attribute& attr);

// b/281863212 enable automatic without Op/AttrAsString.
// We add logging via a wrapper struct in order to respect ODS and avoid
// multiple symbol definitions if MLIR or someone else decides to add ostream
// definitions for the MLIR symbols.
struct LoggableOperation {
  mlir::Operation& v;
  // NOLINTNEXTLINE(google-explicit-constructor)
  LoggableOperation(mlir::Operation& v) : v(v) {}
};
std::ostream& operator<<(std::ostream& o, const LoggableOperation& op);

struct LoggableAttribute {
  mlir::Attribute& v;
  // NOLINTNEXTLINE(google-explicit-constructor)
  LoggableAttribute(mlir::Attribute& v) : v(v) {}
};
std::ostream& operator<<(std::ostream& o, const LoggableAttribute& attr);

struct LoggableStringRef {
  const llvm::StringRef& v;
  // NOLINTNEXTLINE(google-explicit-constructor)
  LoggableStringRef(const llvm::StringRef& v) : v(v) {}
};
std::ostream& operator<<(std::ostream& o, const LoggableStringRef& ref);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_STRING_UTIL_H_
