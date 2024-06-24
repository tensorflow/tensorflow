/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/spmd/shardonnay/utils.h"

#include <cstdint>
#include <functional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/spmd/shardonnay/constants.h"

namespace xla {
namespace sdy {

using ::mlir::ArrayRef;
using ::mlir::Attribute;
using ::mlir::DictionaryAttr;
using ::mlir::NamedAttribute;
using ::mlir::Operation;
using ::mlir::SmallVector;
using ::mlir::StringRef;
using xla::sdy::kFrontendAttributesAttr;

using ::mlir::func::FuncOp;

std::string attributeToString(Attribute attr) {
  std::string out;
  llvm::raw_string_ostream os(out);
  attr.print(os);
  return out;
}

mlir::StringAttr getStringAttribute(mlir::Attribute attr,
                                    mlir::OpBuilder& builder) {
  return builder.getStringAttr(attributeToString(attr));
}

DictionaryAttr getFrontendAttrs(Operation* op) {
  return op->getAttrOfType<DictionaryAttr>(kFrontendAttributesAttr);
}

DictionaryAttr getFuncArgFrontendAttrs(FuncOp funcOp, unsigned int index) {
  return funcOp.getArgAttrOfType<DictionaryAttr>(index,
                                                 kFrontendAttributesAttr);
}

namespace {

SmallVector<NamedAttribute> getExistingFrontendAttributes(
    DictionaryAttr frontendAttributes, StringRef excludedAttribute) {
  SmallVector<NamedAttribute> dictEntries;
  if (!frontendAttributes) {
    return dictEntries;
  }
  for (NamedAttribute entry : frontendAttributes) {
    if (entry.getName() != excludedAttribute) {
      dictEntries.emplace_back(entry);
    }
  }
  return dictEntries;
}

void addFrontendAttribute(SmallVector<NamedAttribute>& existingAttributes,
                          StringRef name, Attribute value) {
  mlir::OpBuilder builder(value.getContext());
  existingAttributes.emplace_back(NamedAttribute(
      builder.getStringAttr(name), getStringAttribute(value, builder)));
}

void removeFrontendAttribute(
    DictionaryAttr frontendAttributes, StringRef attributeName,
    std::function<void(ArrayRef<NamedAttribute>)> setAttr,
    std::function<void()> removeAttr) {
  SmallVector<NamedAttribute> existingAttributes =
      getExistingFrontendAttributes(frontendAttributes, attributeName);
  if (!existingAttributes.empty()) {
    setAttr(existingAttributes);
  } else {
    removeAttr();
  }
}

void setFrontendAttrs(Operation* op, ArrayRef<NamedAttribute> frontendAttrs) {
  return op->setAttr(kFrontendAttributesAttr,
                     DictionaryAttr::get(op->getContext(), frontendAttrs));
}

void setFuncArgFrontendAttrs(FuncOp funcOp, unsigned int index,
                             ArrayRef<NamedAttribute> frontendAttrs) {
  funcOp.setArgAttr(index, kFrontendAttributesAttr,
                    DictionaryAttr::get(funcOp.getContext(), frontendAttrs));
}

}  // namespace

void addFrontendAttribute(Operation* op, StringRef name, Attribute value) {
  SmallVector<NamedAttribute> existingAttributes =
      getExistingFrontendAttributes(getFrontendAttrs(op), "");
  addFrontendAttribute(existingAttributes, name, value);
  setFrontendAttrs(op, existingAttributes);
}

void addFrontendAttribute(FuncOp funcOp, StringRef name, Attribute value,
                          int64_t argNum) {
  SmallVector<NamedAttribute> existingAttributes =
      getExistingFrontendAttributes(getFuncArgFrontendAttrs(funcOp, argNum),
                                    "");
  addFrontendAttribute(existingAttributes, name, value);
  setFuncArgFrontendAttrs(funcOp, argNum, existingAttributes);
}

void removeFrontendAttribute(Operation* op, StringRef attributeName) {
  removeFrontendAttribute(
      getFrontendAttrs(op), attributeName,
      [&](ArrayRef<NamedAttribute> newDict) { setFrontendAttrs(op, newDict); },
      [&]() { op->removeAttr(kFrontendAttributesAttr); });
}

void removeFrontendAttribute(FuncOp funcOp, StringRef attributeName,
                             int64_t argNum) {
  removeFrontendAttribute(
      getFuncArgFrontendAttrs(funcOp, argNum), attributeName,
      [&](ArrayRef<NamedAttribute> newDict) {
        setFuncArgFrontendAttrs(funcOp, argNum, newDict);
      },
      [&]() { funcOp.removeArgAttr(argNum, kFrontendAttributesAttr); });
}

}  // namespace sdy
}  // namespace xla
