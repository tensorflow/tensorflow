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
#include "tensorflow/compiler/mlir/tensorflow/utils/string_util.h"

#include <ostream>
#include <string>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project

namespace tensorflow {

// Return a string form of `op` including debug information.
std::string OpAsString(mlir::Operation& op) {
  std::string out;
  llvm::raw_string_ostream op_stream(out);
  op.print(op_stream, mlir::OpPrintingFlags()
                          .elideLargeElementsAttrs()
                          .assumeVerified()
                          .skipRegions()
                          .printGenericOpForm());
  return out;
}

std::string AttrAsString(mlir::Attribute& attr) {
  std::string out;
  llvm::raw_string_ostream attr_stream(out);
  attr.print(attr_stream);
  return out;
}

std::ostream& operator<<(std::ostream& o, const LoggableOperation& op) {
  return o << OpAsString(op.v);
}

std::ostream& operator<<(std::ostream& o, const LoggableAttribute& attr) {
  return o << AttrAsString(attr.v);
}

std::ostream& operator<<(std::ostream& o, const LoggableStringRef& ref) {
  return o << ref.v.str();
}

}  // namespace tensorflow
