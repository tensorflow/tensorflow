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

#include "mlir-hlo/Dialect/gml_st/transforms/tiling_interface.h"

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/tiling_interface.cc.inc"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace gml_st {

namespace {
struct ElementWiseTilingInterface
    : public TilingInterface::ExternalModel<ElementWiseTilingInterface,
                                            mlir::mhlo::AddOp> {
  Value tile(Operation* op, MaterializeOp materialize,
             OpBuilder& builder) const {
    if (materialize.subset().getType().isa<PointType>()) {
      Location loc = materialize.getLoc();
      // Push the materialize to the arguments and replace op by scalar version.
      auto addOp = cast<mhlo::AddOp>(op);
      auto newLhs =
          builder.create<MaterializeOp>(loc, addOp.lhs(), materialize.subset());
      auto newRhs =
          builder.create<MaterializeOp>(loc, addOp.rhs(), materialize.subset());
      return builder.create<arith::AddFOp>(loc, newLhs, newRhs);
    }
    return {};
  }
};

}  // namespace

void registerTilingInterfaceExternalModels(DialectRegistry& registry) {
  registry.insert<mhlo::MhloDialect>();
  registry.addOpInterface<mhlo::AddOp, ElementWiseTilingInterface>();
}

}  // namespace gml_st
}  // namespace mlir
