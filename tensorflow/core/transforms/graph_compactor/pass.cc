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

#include <memory>
#include <string>
#include <utility>

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/interfaces.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/transforms/pass_detail.h"

namespace mlir {
namespace tfg {

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
class NameCompressPass : public NameCompressBase<NameCompressPass> {
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
      assert(func.arg_attrs().hasValue() && "expected argument attributes");
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
      assert(func.res_attrs().hasValue() && "expected result attributes");
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
}  // namespace tfg
}  // namespace mlir
