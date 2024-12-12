/* Copyright 2024 The OpenXLA Authors.

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

#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "mlir/IR/DialectImplementation.h"  // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h"  // IWYU pragma: keep
#include "mlir/Transforms/InliningUtils.h"
#include "xla/backends/gpu/codegen/ir/xla_gpu_ops.h"

// The order of these includes is important.
#include "xla/backends/gpu/codegen/ir/xla_gpu_enums.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "xla/backends/gpu/codegen/ir/xla_gpu_attrs.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "xla/backends/gpu/codegen/ir/xla_gpu_types.cc.inc"

namespace xla {
namespace gpu {
namespace {


struct XlaGpuOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(mlir::Attribute attr,
                       mlir::raw_ostream& os) const final {
    if (llvm::isa<LayoutAttr>(attr)) {
      os << "layout";
      return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }
};

}  // namespace

void XlaGpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "xla/backends/gpu/codegen/ir/xla_gpu_ops.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xla/backends/gpu/codegen/ir/xla_gpu_attrs.cc.inc"
      >();
  addInterfaces<XlaGpuOpAsmDialectInterface>();
  addTypes<
#define GET_TYPEDEF_LIST
#include "xla/backends/gpu/codegen/ir/xla_gpu_types.cc.inc"
      >();
}

}  // namespace gpu
}  // namespace xla
