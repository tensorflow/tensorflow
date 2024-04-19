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

#include "tensorflow/compiler/mlir/tf2xla/internal/utils/dialect_detection_utils.h"

#include <set>
#include <string>

#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project

namespace tensorflow {
namespace tf2xla {
namespace internal {

bool IsInBridgeAcceptableDialects(mlir::Operation* op) {
  const std::set<std::string> kBuiltinNamespaces = {"func", "return",
                                                    "builtin"};
  const std::set<std::string> kBridgeAcceptableNamespaces = {"tf", "tf_device"};
  bool isInDefaulNamespaces =
      kBuiltinNamespaces.find(op->getDialect()->getNamespace().str()) !=
      kBuiltinNamespaces.end();
  bool isInBridgeAcceptableNamespaces =
      kBridgeAcceptableNamespaces.find(
          op->getDialect()->getNamespace().str()) !=
      kBridgeAcceptableNamespaces.end();
  return isInDefaulNamespaces || isInBridgeAcceptableNamespaces;
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
