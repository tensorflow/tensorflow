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

#include "tensorflow/core/transforms/graph_compactor/pass.h"

#include <iterator>
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/BitVector.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/importexport/convert_attributes.h"
#include "tensorflow/core/ir/interfaces.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/tf_op_registry.h"
#include "tensorflow/core/platform/statusor.h"

namespace mlir {
namespace tfg {

#define GEN_PASS_DEF_ADDDEFAULTATTRS
#define GEN_PASS_DEF_NAMECOMPRESS
#define GEN_PASS_DEF_STRIPDEFAULTATTRS
#include "tensorflow/core/transforms/passes.h.inc"

// Encode an unsigned integer in as few characters as possible to a string that
// is still a valid TensorFlow node name. The regex for valid names, according
// to `NodeDef`, is "[A-Za-z0-9.][A-Za-z0-9_>./]*"
//
// The valid characters are provided in the two arrays `first_valid_chars` and
// `trailing_valid_chars`.
static void EncodeName(unsigned counter, std::string &output,
                       ArrayRef<char> first_valid_chars,
                       ArrayRef<char> trailing_valid_chars) {
  assert(!first_valid_chars.empty() && !trailing_valid_chars.empty());
  unsigned rem = counter % first_valid_chars.size();
  counter /= first_valid_chars.size();
  output.push_back(first_valid_chars[rem]);
  while (counter > 0) {
    --counter;
    rem = counter % trailing_valid_chars.size();
    counter /= trailing_valid_chars.size();
    output.push_back(trailing_valid_chars[rem]);
  }
}

// Encode an unsigned integer to a valid TensorFlow node name.
static void EncodeName(unsigned counter, std::string &output) {
  // The alphabet of valid characters, but the last 3 are only valid in trailing
  // characters.
  static constexpr char valid_chars[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._>/";
  // Sanity check: all alphanumeric characters, four special characters, and a
  // null terminator.
  constexpr unsigned valid_first_chars = 26 * 2 + 10 + 1;
  constexpr unsigned valid_trailing_chars = valid_first_chars + 3;
  static_assert(sizeof(valid_chars) == valid_trailing_chars + 1,
                "alphabet sanity check");
  EncodeName(counter, output,
             llvm::makeArrayRef(valid_chars, valid_first_chars),
             llvm::makeArrayRef(valid_chars, valid_trailing_chars));
}

namespace {
class NameCompressPass : public impl::NameCompressBase<NameCompressPass> {
 public:
  LogicalResult initialize(MLIRContext *context) override {
    dialect_ = context->getOrLoadDialect<TFGraphDialect>();
    empty_dict_ = DictionaryAttr::get(context);
    return success();
  }

  void runOnOperation() override {
    GraphFuncOp func = getOperation();

    Builder b(&getContext());
    unsigned counter = 0;
    std::string name;
    const auto encode_new_name = [&name, &b, &counter] {
      name.clear();
      EncodeName(counter++, name);
      return b.getStringAttr(name);
    };

    // Rename the arguments and results.
    NamedAttrList attrs = func->getAttrDictionary();
    if (func.getNumArguments()) {
      assert(func.arg_attrs().has_value() && "expected argument attributes");
      SmallVector<Attribute> arg_attrs;
      arg_attrs.reserve(func.getNumArguments());
      // Iterate over the function arguments, skipping the control tokens.
      for (int i = 0, e = func.getNumArguments(); i != e; i += 2) {
        NamedAttrList attrs = func.arg_attrsAttr()[i].cast<DictionaryAttr>();
        attrs.set(dialect_->getTfgNameAttrIdentifier(), encode_new_name());
        arg_attrs.append({attrs.getDictionary(&getContext()), empty_dict_});
      }
      attrs.set(func.arg_attrsAttrName(), b.getArrayAttr(arg_attrs));
    }
    if (func.getNumResults()) {
      assert(func.res_attrs().has_value() && "expected result attributes");
      SmallVector<Attribute> res_attrs;
      res_attrs.reserve(func.getNumResults());
      for (NamedAttrList attrs :
           func.res_attrsAttr().getAsRange<DictionaryAttr>()) {
        attrs.set(dialect_->getTfgNameAttrIdentifier(), encode_new_name());
        res_attrs.push_back(attrs.getDictionary(&getContext()));
      }
      attrs.set(func.res_attrsAttrName(), b.getArrayAttr(res_attrs));
    }
    if (func.getNumArguments() || func.getNumResults()) {
      func->setAttrs(attrs.getDictionary(&getContext()));
    }

    // Rename the control results.
    ReturnOp terminator = cast<ReturnOp>(func.getBody()->getTerminator());
    ArrayAttr control_attrs = terminator.control_ret_attrs();
    if (!attrs.empty()) {
      SmallVector<Attribute> control_ret_attrs;
      control_ret_attrs.reserve(control_attrs.size());
      for (NamedAttrList attrs : control_attrs.getAsRange<DictionaryAttr>()) {
        attrs.set(dialect_->getTfgNameAttrIdentifier(), encode_new_name());
        control_ret_attrs.push_back(attrs.getDictionary(&getContext()));
      }
      terminator.control_ret_attrsAttr(b.getArrayAttr(control_ret_attrs));
    }

    // Rename all non-intrisic operations.
    func.walk([this, &encode_new_name](Operation *op) {
      if (op->hasTrait<OpTrait::IntrinsicOperation>()) return;
      op->setAttr(dialect_->getNameAttrIdentifier(), encode_new_name());
    });
  }

 private:
  // An instance of the TFG dialect for accessing cached identifiers.
  TFGraphDialect *dialect_;
  // An instance of the empty dictionary attribute.
  DictionaryAttr empty_dict_;
};
}  // namespace

std::unique_ptr<Pass> CreateNameCompressPass() {
  return std::make_unique<NameCompressPass>();
}

namespace {
class StripDefaultAttrsPass
    : public impl::StripDefaultAttrsBase<StripDefaultAttrsPass> {
 public:
  LogicalResult initialize(MLIRContext *context) override {
    // Initialize the pass by getting a registered instance of the TensorFlow
    // operation registry. If no instance was registered, this pass will fail.
    dialect_ = context->getOrLoadDialect<TFGraphDialect>();
    registry_ = nullptr;
    if (auto registry_interface =
            dialect_->getRegisteredInterface<TensorFlowOpRegistryInterface>()) {
      registry_ = registry_interface->GetRegistry();
    }
    return success(registry_);
  }

  void runOnOperation() override {
    WalkResult result = getOperation()->walk([&](Operation *op) {
      // Ignore intrinsic operations.
      if (op->hasTrait<OpTrait::IntrinsicOperation>())
        return WalkResult::advance();

      // If removing default-valued attributes failed (attribute conversion
      // error), bail out.
      if (failed(removeDefaultValuedAttrs(op))) return WalkResult::interrupt();

      return WalkResult::advance();
    });

    // If the pass failed on any operation, signal failure.
    if (result.wasInterrupted()) return signalPassFailure();
  }

 private:
  // Remove attributes from the operation equal to their default values
  // according to the TensorFlow op registry.
  LogicalResult removeDefaultValuedAttrs(Operation *op);

  // The TFG dialect instance.
  TFGraphDialect *dialect_;
  // The TensorFlow op registry to query for default-valued attributes.
  const tensorflow::OpRegistry *registry_;
};
}  // namespace

LogicalResult StripDefaultAttrsPass::removeDefaultValuedAttrs(Operation *op) {
  const tensorflow::OpRegistrationData *op_reg_data =
      registry_->LookUp(op->getName().stripDialect().str());
  // Ignore unregistered ops.
  if (!op_reg_data) return success();

  // Find the attributes to remove.
  ArrayRef<NamedAttribute> attrs = op->getAttrs();
  llvm::BitVector indices_to_remove(attrs.size());
  Builder b(&getContext());
  for (const tensorflow::OpDef::AttrDef &attr : op_reg_data->op_def.attr()) {
    // Ignore attributes without default values.
    if (!attr.has_default_value()) continue;
    auto it =
        ::mlir::impl::findAttrSorted(attrs.begin(), attrs.end(), attr.name());
    // Ignore default-valued attributes that are already missing.
    if (!it.second) continue;
    // Convert the TensorFlow attribute value and compare it to the MLIR
    // attribute.
    tensorflow::StatusOr<Attribute> maybe_attr =
        ConvertAttributeValue(attr.default_value(), b);
    if (!maybe_attr.ok())
      return op->emitError(maybe_attr.status().error_message());
    if (maybe_attr.value() == it.first->getValue())
      indices_to_remove.set(std::distance(attrs.begin(), it.first));
  }
  if (indices_to_remove.none()) return success();

  // Construct and set the new attributes.
  SmallVector<NamedAttribute> new_attrs;
  new_attrs.reserve(attrs.size());
  for (auto &it : llvm::enumerate(attrs)) {
    if (indices_to_remove.test(it.index())) continue;
    new_attrs.push_back(it.value());
  }
  op->setAttrs(DictionaryAttr::getWithSorted(&getContext(), new_attrs));

  return success();
}

std::unique_ptr<Pass> CreateStripDefaultAttrsPass() {
  return std::make_unique<StripDefaultAttrsPass>();
}

namespace {
class AddDefaultAttrsPass
    : public impl::AddDefaultAttrsBase<AddDefaultAttrsPass> {
 public:
  LogicalResult initialize(MLIRContext *context) override {
    // Initialize the pass by getting a registered instance of the TensorFlow
    // operation registry. If no instance was registered, this pass will fail.
    dialect_ = context->getOrLoadDialect<TFGraphDialect>();
    registry_ = nullptr;
    if (auto registry_interface =
            dialect_->getRegisteredInterface<TensorFlowOpRegistryInterface>()) {
      registry_ = registry_interface->GetRegistry();
    }
    return success(registry_);
  }

  void runOnOperation() override {
    WalkResult result = getOperation()->walk([&](Operation *op) {
      // Ignore intrinsic operations.
      if (op->hasTrait<OpTrait::IntrinsicOperation>())
        return WalkResult::advance();

      // If removing default-valued attributes failed (attribute conversion
      // error), bail out.
      if (failed(addDefaultValuedAttrs(op))) return WalkResult::interrupt();

      return WalkResult::advance();
    });

    // If the pass failed on any operation, signal failure.
    if (result.wasInterrupted()) return signalPassFailure();
  }

 private:
  // Remove attributes from the operation equal to their default values
  // according to the TensorFlow op registry.
  LogicalResult addDefaultValuedAttrs(Operation *op);

  // The TFG dialect instance.
  TFGraphDialect *dialect_;
  // The TensorFlow op registry to query for default-valued attributes.
  const tensorflow::OpRegistry *registry_;
};
}  // namespace

LogicalResult AddDefaultAttrsPass::addDefaultValuedAttrs(Operation *op) {
  const tensorflow::OpRegistrationData *op_reg_data =
      registry_->LookUp(op->getName().stripDialect().str());
  // Ignore unregistered ops.
  if (!op_reg_data) return success();

  // Ignore operations with no default-valued attributes.
  if (llvm::all_of(op_reg_data->op_def.attr(),
                   [](const auto &attr) { return !attr.has_default_value(); }))
    return success();

  // Add missing default-valued attributes
  Builder b(&getContext());
  NamedAttrList attrs = op->getAttrDictionary();
  for (const auto &attr : op_reg_data->op_def.attr()) {
    // Ignore attributes without default values.
    if (!attr.has_default_value()) continue;
    // Ignore default-valued attributes that are present.
    if (attrs.get(attr.name())) continue;
    // Convert the TensorFlow attribute value and set it.
    tensorflow::StatusOr<Attribute> maybe_attr =
        ConvertAttributeValue(attr.default_value(), b);
    if (!maybe_attr.ok())
      return op->emitError(maybe_attr.status().error_message());
    attrs.set(attr.name(), maybe_attr.value());
  }
  op->setAttrs(attrs.getDictionary(&getContext()));

  return success();
}

std::unique_ptr<Pass> CreateAddDefaultAttrsPass() {
  return std::make_unique<AddDefaultAttrsPass>();
}

}  // namespace tfg
}  // namespace mlir
