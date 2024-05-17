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

#include <iterator>
#include <string>
#include <utility>

#include "absl/strings/str_split.h"
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
  if (!replicate_attr && !compile_attr) return success();
  if (!replicate_attr || !compile_attr)
    return op.emitOpError() << "is expected to have either both or none of '"
                            << kReplicationInfoAttr << "' and '"
                            << kCompileDeviceTypeAttr << "' attributes.";
  if (replicate_attr.getValue().empty())
    return op.emitOpError()
           << "has an empty '" << kReplicationInfoAttr << "' attribute.";
  if (failed(IsValidDeviceTypeOrEmpty(compile_attr))) {
    return op.emitOpError() << "has invalid '" << kCompileDeviceTypeAttr
                            << "' value '" << compile_attr.getValue() << "'";
  }
  return success();
}

LogicalResult IsValidDeviceTypeOrEmpty(StringAttr device_attr) {
  auto value = device_attr.getValue();
  // TODO(b/229028654): Remove string conversion once we have C++17.
  absl::string_view device_type(value.data(), value.size());
  // Device type may be empty for some ops, e.g. tf.PartitionedCall.
  auto it = std::find(kValidDeviceTypes.begin(), kValidDeviceTypes.end(),
                      device_type);
  if (it == kValidDeviceTypes.end()) return failure();
  return success();
}

LogicalResult ParseParallelExecutionIds(Operation* op,
                                        ParallelExecutionIdPairs& id_pairs) {
  auto attr = op->getAttrOfType<StringAttr>(kParallelExecAnnotation);
  if (!attr) return success();

  // ID pairs are separated by `,`.
  llvm::SmallVector<std::string, 8> str_list =
      absl::StrSplit(attr.getValue().str(), ',', absl::SkipWhitespace());
  id_pairs.reserve(str_list.size());
  for (const std::string& str : str_list) {
    // IDs of one pair are separated by `:`.
    llvm::SmallVector<std::string, 8> id_pair = absl::StrSplit(str, ':');

    // Check for malformed IDs.
    if (id_pair.size() != 2) return failure();
    if (id_pair[0].empty() || id_pair[1].empty()) return failure();

    auto is_digit = [](char c) { return absl::ascii_isdigit(c); };
    const std::string& group_id = id_pair[0];
    if (group_id[0] != 'p' && group_id[0] != 'r') return failure();
    if (!std::all_of(std::next(group_id.begin()), group_id.end(), is_digit)) {
      return failure();
    }
    const std::string& branch_id = id_pair[1];
    if (!std::all_of(branch_id.begin(), branch_id.end(), is_digit)) {
      return failure();
    }
    id_pairs.push_back(std::make_pair(id_pair[0], id_pair[1]));
  }
  return success();
}

}  // namespace TF
}  // namespace mlir
