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

#include "xla/codegen/xtile/ir/xtile_dialect.h"

#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/Transforms/InliningUtils.h"
#include "xla/codegen/xtile/ir/xtile_attrs.h"  // IWYU pragma: keep
#include "xla/codegen/xtile/ir/xtile_ops.h"  // IWYU pragma: keep

// Include the auto-generated implementation file.
#include "xla/codegen/xtile/ir/xtile_dialect.cc.inc"

#define GET_ATTRDEF_CLASSES
#include "xla/codegen/xtile/ir/xtile_attrs.cc.inc"

namespace xla::xtile {

namespace {

struct XTileInlinerInterface final : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // We allow all callables to be inlined.
  bool isLegalToInline(mlir::Operation* call, mlir::Operation* callable,
                       bool wouldBeCloned) const override {
    return true;
  }

  // We allow any op from the xla dialect to be inlined.
  bool isLegalToInline(mlir::Operation* op, mlir::Region* dest,
                       bool wouldBeCloned,
                       mlir::IRMapping& valueMapping) const override {
    return true;
  }
  // We allow any ops to be inlined into any region.
  bool isLegalToInline(mlir::Region* dest, mlir::Region* src,
                       bool wouldBeCloned,
                       mlir::IRMapping& valueMapping) const override {
    return true;
  }
};

}  // namespace

void XTileDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xla/codegen/xtile/ir/xtile_ops.cc.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "xla/codegen/xtile/ir/xtile_attrs.cc.inc"
      >();

  addInterfaces<XTileInlinerInterface>();
}

}  // namespace xla::xtile
