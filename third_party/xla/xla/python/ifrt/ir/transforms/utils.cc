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

#include "xla/python/ifrt/ir/transforms/utils.h"

#include "absl/log/check.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"

namespace xla {
namespace ifrt {

mlir::func::FuncOp GetMainFunction(mlir::ModuleOp module) {
  mlir::func::FuncOp func =
      mlir::dyn_cast_or_null<mlir::func::FuncOp>(module.lookupSymbol("main"));
  CHECK(func);
  return func;
}

bool IsReshard(xla::ifrt::IfrtArrayType from, xla::ifrt::IfrtArrayType to) {
  if (from.getShape() == to.getShape() &&
      from.getShardingAttr() == to.getShardingAttr() &&
      from.getDevices().size() == to.getDevices().size()) {
    return false;
  }
  return true;
}

}  // namespace ifrt
}  // namespace xla
