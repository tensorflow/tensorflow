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

#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

#include <algorithm>

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/tf2xla/tf2xla_defs.h"

namespace mlir {
namespace TF {

using ::tensorflow::kValidDeviceTypes;

LogicalResult HasValidCompilationAndReplicationAttributes(Operation& op) {
  auto replicate_attr = op.getAttrOfType<StringAttr>(kReplicationInfoAttr);
  auto compile_attr = op.getAttrOfType<StringAttr>(kCompileDeviceTypeAttr);
  if (replicate_attr && !compile_attr) {
    return op.emitOpError()
           << "has '" << kReplicationInfoAttr << "' attribute but not '"
           << kCompileDeviceTypeAttr << "' attribute which is unsupported";
  }
  if (replicate_attr && replicate_attr.getValue().empty()) {
    return op.emitOpError()
           << "has an empty '" << kReplicationInfoAttr << "' attribute";
  }
  if (compile_attr) {
    auto value = compile_attr.getValue();
    // TODO(b/229028654): Remove string conversion once we have C++17.
    absl::string_view device_type(value.data(), value.size());
    auto it = std::find(kValidDeviceTypes.begin(), kValidDeviceTypes.end(),
                        device_type);
    if (it == kValidDeviceTypes.end()) {
      return op.emitOpError() << "has invalid '" << kCompileDeviceTypeAttr
                              << "' value '" << compile_attr.getValue() << "'";
    }
  }
  return success();
}

}  // namespace TF
}  // namespace mlir
