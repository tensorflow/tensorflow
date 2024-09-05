/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfr/utils/utils.h"

#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfr/ir/tfr_ops.h"

namespace mlir {
namespace TFR {
namespace {

// TODO(b/174692018): Use the official allowlist of the unregistered attrs.
const llvm::StringSet<>& GetAllowedAttributes() {
  static auto* const ops = new llvm::StringSet<>({"device", "_tpu_replicate"});
  return *ops;
}

// Some TFL optional attributes may not appear in their corresponding TF op
// attributes.
const llvm::StringSet<>& GetOptionalAttributes() {
  static auto* const ops =
      new llvm::StringSet<>({"asymmetric_quantize_inputs"});
  return *ops;
}

void CollectAllowedAttrs(CallOp src, NamedAttrList* attrs) {
  for (auto& attr : src->getAttrs()) {
    if (GetAllowedAttributes().contains(attr.getName().strref())) {
      attrs->append(attr);
    }
  }
}

// Adds `attrs` to all the operations between `begin` and `end` in the same
// block. Does not include `end`.
void AddAttributesInSameBlock(Block::iterator begin, Block::iterator end,
                              const NamedAttrList& attrs) {
  for (Block::iterator it = begin; it != end; ++it) {
    for (auto& attr : attrs) {
      it->setAttr(attr.getName(), attr.getValue());
    }
  }
}

// Adds `attrs` to all the operations between `begin` and `end`. Does not
// include `end`. The operations might be across multiple  blocks.
void AddAttributes(Block::iterator begin, Block::iterator end,
                   const NamedAttrList& attrs) {
  if (begin->getBlock() == end->getBlock()) {
    AddAttributesInSameBlock(begin, end, attrs);
  } else {
    Region::iterator begin_block = Region::iterator(begin->getBlock());
    Region::iterator end_block = Region::iterator(end->getBlock());
    AddAttributesInSameBlock(begin, begin_block->end(), attrs);
    for (Region::iterator it = ++begin_block; it != end_block; ++it) {
      AddAttributesInSameBlock(it->begin(), it->end(), attrs);
    }
  }
}

}  // namespace

std::string GetComposeFuncName(StringRef tf_op_name) {
  std::string compose_func_name;
  for (int i = 0; i < tf_op_name.size(); ++i) {
    if (tf_op_name[i] == '_') {
      // The field name must not contain "_"s. "_Arg" and "_RetVal" are special
      // op names and we can return empty string to skip the decomposition.
      return {};
    }
    if (tf_op_name[i] == '.') {
      compose_func_name.push_back('_');
    } else if (tf_op_name[i] >= 'A' && tf_op_name[i] <= 'Z') {
      compose_func_name.push_back('_');
      compose_func_name.push_back(tf_op_name[i] + 'a' - 'A');
    } else {
      compose_func_name.push_back(tf_op_name[i]);
    }
  }
  return compose_func_name;
}

std::string GetTFOpName(StringRef compose_func_name) {
  std::string tf_op_name;
  bool after_underscore = false;
  for (int i = 0; i < compose_func_name.size(); ++i) {
    if (compose_func_name[i] >= 'A' && compose_func_name[i] <= 'Z') {
      // The field name must not contain uppercase letters.
      return {};
    }
    if (after_underscore) {
      if (compose_func_name[i] >= 'a' && compose_func_name[i] <= 'z') {
        tf_op_name.push_back(compose_func_name[i] + 'A' - 'a');
        after_underscore = false;
      } else {
        // The character after a "_" must be a lowercase letter.
        return {};
      }
    } else if (compose_func_name[i] == '_') {  // first time visit '_'
      if (i + 1 < compose_func_name.size() && compose_func_name[i + 1] == '_') {
        tf_op_name.push_back('.');
        i++;
      }
      after_underscore = true;
    } else {
      tf_op_name.push_back(compose_func_name[i]);
    }
  }
  if (after_underscore) {
    // Trailing "_".
    return {};
  }
  return tf_op_name;
}

LogicalResult ValidateAttrs(Operation* src, const StringSet<>& registered) {
  for (auto& attr : src->getAttrs()) {
    StringRef attr_name = attr.getName().strref();

    if (!registered.contains(attr_name) &&
        !(GetAllowedAttributes().contains(attr_name) ||
          GetOptionalAttributes().contains(attr_name))) {
      src->emitError("Denied unregistered attribute was found: " + attr_name);
      return failure();
    }
  }
  return success();
}

LogicalResult CopyAllowedUnregisteredAttrs(Operation* src, CallOp dst,
                                           const StringSet<>& registered) {
  for (auto& attr : src->getAttrs()) {
    StringRef attr_name = attr.getName().strref();
    // Skip the registered or optional attribute.
    if (registered.contains(attr_name) ||
        GetOptionalAttributes().contains(attr_name))
      continue;

    // Unregistered attribute.
    if (GetAllowedAttributes().contains(attr_name)) {
      dst->setAttr(attr.getName(), attr.getValue());
    } else {
      src->emitError("Denied unregistered attribute was found: " + attr_name);
      return failure();
    }
  }
  return success();
}

LogicalResult CopyNonSymbolRefAttrs(CallOp src, Operation* dst) {
  NamedAttrList attrs;
  CollectAllowedAttrs(src, &attrs);

  for (auto& attr : attrs) {
    dst->setAttr(attr.getName(), attr.getValue());
  }

  return success();
}

void PropagateAttrsToOperations(CallOp src, Block::iterator begin,
                                Block::iterator end) {
  // Find all the attributes in the call op. These attributes are not in the
  // op definition, so needs to be propagated to all the target ops.
  NamedAttrList attrs;
  CollectAllowedAttrs(src, &attrs);

  // Add all the attributes to the operations in the range.
  if (!attrs.empty()) {
    AddAttributes(begin, end, attrs);
  }
}

}  // namespace TFR
}  // namespace mlir
