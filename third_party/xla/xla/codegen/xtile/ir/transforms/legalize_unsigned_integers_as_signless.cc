/* Copyright 2025 The OpenXLA Authors.

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

#include <optional>

#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"  // IWYU pragma: keep

namespace xla::xtile {

#define GEN_PASS_DEF_LEGALIZEUNSIGNEDINTEGERSASSIGNLESSPASS
#include "xla/codegen/xtile/ir/transforms/passes.h.inc"

namespace {

using mlir::AttrTypeReplacer;
using mlir::cast;
using mlir::DenseElementsAttr;
using mlir::dyn_cast;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::MLIRContext;
using mlir::Type;

struct LegalizeUnsignedIntegersAsSignlessPass
    : public impl::LegalizeUnsignedIntegersAsSignlessPassBase<
          LegalizeUnsignedIntegersAsSignlessPass> {
  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext* context = &getContext();

    AttrTypeReplacer replacer;

    // 1. Map Unsigned Integers to Signless Integers
    replacer.addReplacement([&](Type type) -> std::optional<Type> {
      if (auto int_type = dyn_cast<IntegerType>(type)) {
        if (int_type.isUnsignedInteger()) {
          return IntegerType::get(context, int_type.getWidth());
        }
      }
      return std::nullopt;
    });

    // 2. Map Unsigned Integers Attributes to Signless Integer Attributes
    replacer.addReplacement(
        [&](mlir::Attribute attr) -> std::optional<mlir::Attribute> {
          // Handle single integer constants
          if (auto int_attr = dyn_cast<IntegerAttr>(attr)) {
            if (int_attr.getType().isUnsignedInteger()) {
              auto unsigned_int_type =
                  mlir::cast<IntegerType>(int_attr.getType());
              return IntegerAttr::get(
                  IntegerType::get(context, unsigned_int_type.getWidth()),
                  int_attr.getValue());
            }
          }

          // Handle Dense Elements (Tensors/Vectors)
          if (auto dense_attr = dyn_cast<DenseElementsAttr>(attr)) {
            if (dense_attr.getElementType().isUnsignedInteger()) {
              auto unsigned_int_type =
                  mlir::cast<IntegerType>(dense_attr.getElementType());
              Type new_element_type =
                  IntegerType::get(context, unsigned_int_type.getWidth());
              mlir::DenseElementsAttr new_dense_attr = dense_attr.mapValues(
                  new_element_type, [&](llvm::APInt val) { return val; });

              return new_dense_attr;
            }
          }

          return std::nullopt;
        });

    // 3. Execute the global replacement
    replacer.recursivelyReplaceElementsIn(module,
                                          /*replaceAttrs=*/true,
                                          /*replaceLocs=*/false,
                                          /*replaceTypes=*/true);
  }
};

}  // namespace

}  // namespace xla::xtile
