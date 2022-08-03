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

#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface_impl.h"

#include <tuple>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace gml_st {

namespace {

bool isTransposeOrElementwise(linalg::GenericOp genericOp) {
  // Only consider all-parallel `linalg.generic` ops with a unique result and
  // tensor semantics for fusion.
  if (!genericOp.hasTensorSemantics() || genericOp.outputs().size() != 1 ||
      llvm::any_of(genericOp.iterator_types(), [](Attribute attr) {
        return !mlir::isParallelIterator(attr);
      })) {
    return false;
  }

  // Fuse if op is transpose (or element-wise).
  if (llvm::all_of(genericOp.indexing_maps(), [](Attribute attr) {
        auto map = attr.cast<AffineMapAttr>().getAffineMap();
        assert((!map.isIdentity() || map.isPermutation()) &&
               "expect identity maps to be considered a permutation");
        return map.isPermutation();
      })) {
    return true;
  }

  return false;
}

struct LinalgGenericFusionInterface
    : public FusionInterface::ExternalModel<LinalgGenericFusionInterface,
                                            linalg::GenericOp> {
  Value fuse(Operation* op, Location loc, Value subset,
             OpBuilder& builder) const {
    auto genericOp = llvm::cast<linalg::GenericOp>(op);

    // Only fuse transpose (or element-wise) `linalg.generic` ops.
    if (!isTransposeOrElementwise(genericOp)) return {};

    // Materialze fused operands.
    SmallVector<Value> subOperands;
    subOperands.reserve(genericOp.getNumInputs());
    for (const auto& it :
         llvm::zip(genericOp.inputs(), genericOp.getIndexingMapsArray())) {
      Value operand;
      AffineMap map;
      std::tie(operand, map) = it;

      // Create subset for the current operand from the result subset.
      assert(map.isPermutation() && "expect permutation (or identity) map");
      Value operandSubset = subset;

      // Transpose operand subset if needed.
      if (!map.isIdentity()) {
        const unsigned int rank = map.getNumResults();
        SmallVector<int64_t> permutation;
        permutation.reserve(rank);
        for (int i = 0; i < static_cast<int>(rank); ++i) {
          permutation.push_back(map.getPermutedPosition(i));
        }
        operandSubset = builder.create<TransposeDimsOp>(
            loc, operandSubset,
            DenseI64ArrayAttr::get(builder.getContext(), permutation));
      }

      // Materialize subset of current operand.
      subOperands.push_back(
          builder.create<MaterializeOp>(loc, operand, operandSubset));
    }

    // Materialize the tiled output.
    Value output = genericOp.outputs().front();
    subOperands.push_back(builder.create<MaterializeOp>(loc, output, subset));

    const Type subsetTy = subset.getType();
    return llvm::TypeSwitch<Type, Value>(subsetTy)
        .Case([&](TileType tileTy) -> Value {
          auto outputTy = output.getType().cast<RankedTensorType>();
          auto subResultTy = RankedTensorType::get(tileTy.getShape(),
                                                   outputTy.getElementType());

          // Materialize tiled `linalg.generic` op.
          linalg::LinalgOp linalgOp = genericOp;
          return linalgOp.clone(builder, loc, subResultTy, subOperands)
              ->getResults()
              .front();
        })
        .Case([&](PointType) -> Value {
          // Create scalar computation by copying from the `linalg.generic`
          // body.
          BlockAndValueMapping bvm;
          Block* block = genericOp.getBody();
          for (const auto& it : llvm::zip(block->getArguments(), subOperands)) {
            bvm.map(std::get<0>(it), std::get<1>(it));
          }
          for (auto& it : block->without_terminator()) builder.clone(it, bvm);
          auto innerResults = block->getTerminator()->getOperands();
          assert(innerResults.size() == 1 && "expect unique inner result");
          return bvm.lookup(innerResults.front());
        })
        .Default([](Type) -> Value { return {}; });
  }
};

}  // namespace

void registerFusionInterfaceExternalModels(DialectRegistry& registry) {
  registry.insert<linalg::LinalgDialect>();
  registry.addExtension(+[](MLIRContext* ctx, linalg::LinalgDialect*) {
    linalg::GenericOp::attachInterface<LinalgGenericFusionInterface>(*ctx);
  });
}

}  // namespace gml_st
}  // namespace mlir
