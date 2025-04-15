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

#include "xla/service/spmd/shardy/extensions/mhlo_extensions.h"

#include <cstdint>

#include "absl/log/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/enums.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mhlo = ::mlir::mhlo;

namespace xla {
namespace sdy {

namespace {

using ::mlir::ArrayRef;
using ::mlir::sdy::FactorType;
using ::mlir::sdy::kNullDim;
using ::mlir::sdy::OpShardingRuleBuilder;

struct CopyShardingRuleOpInterface
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          CopyShardingRuleOpInterface, mhlo::CopyOp> {
  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation* op) const {
    return OpShardingRuleBuilder::buildPointwise(op);
  }
};

enum RaggedDotMode {
  // Ragged non-contracting (m): [b,m,k], [g,b,k,n], [b,g] -> [b,m,n].
  kNonContracting,
  // Ragged contracting (k):     [b,m,k], [b,k,n],   [b,g] -> [g,b,m,n].
  kContracting,
  // Ragged batch (b):           [b,m,k], [b,k,n],   [g]   -> [b,m,n].
  kBatch,
};

struct RaggedDotShardingRuleOpInterface
    : public mlir::sdy::ShardingRuleOpInterface::ExternalModel<
          RaggedDotShardingRuleOpInterface, mhlo::RaggedDotOp> {
  mlir::sdy::OpShardingRuleAttr getShardingRule(mlir::Operation* op) const {
    mhlo::RaggedDotOp raggedDot = llvm::cast<mhlo::RaggedDotOp>(op);
    mhlo::RaggedDotDimensionNumbersAttr raggedDotDimNumbers =
        raggedDot.getRaggedDotDimensionNumbers();
    mhlo::DotDimensionNumbersAttr dotDimNumbers =
        raggedDotDimNumbers.getDotDimensionNumbers();

    ArrayRef<int64_t> lhsBatchingDims =
        dotDimNumbers.getLhsBatchingDimensions();
    ArrayRef<int64_t> rhsBatchingDims =
        dotDimNumbers.getRhsBatchingDimensions();
    ArrayRef<int64_t> lhsContractingDims =
        dotDimNumbers.getLhsContractingDimensions();
    ArrayRef<int64_t> rhsContractingDims =
        dotDimNumbers.getRhsContractingDimensions();

    ArrayRef<int64_t> lhsRaggedDims =
        raggedDotDimNumbers.getLhsRaggedDimensions();
    CHECK_EQ(lhsRaggedDims.size(), 1);
    int64_t lhsRaggedDim = lhsRaggedDims.front();
    ArrayRef<int64_t> rhsGroupDims =
        raggedDotDimNumbers.getRhsGroupDimensions();

    RaggedDotMode mode;
    if (llvm::is_contained(lhsContractingDims, lhsRaggedDim)) {
      CHECK(rhsGroupDims.empty());
      mode = RaggedDotMode::kContracting;
    } else if (llvm::is_contained(lhsBatchingDims, lhsRaggedDim)) {
      CHECK(rhsGroupDims.empty());
      mode = RaggedDotMode::kBatch;
    } else {
      CHECK_EQ(rhsGroupDims.size(), 1);
      mode = RaggedDotMode::kNonContracting;
    }

    OpShardingRuleBuilder builder(raggedDot);

    mlir::RankedTensorType lhsType = raggedDot.getLhs().getType();
    mlir::RankedTensorType rhsType = raggedDot.getRhs().getType();
    mlir::RankedTensorType groupSizesType = raggedDot.getGroupSizes().getType();

    int64_t groupSizesDim = 0;
    int64_t outputDim = (mode == RaggedDotMode::kContracting) ? 1 : 0;

    // batching dimensions
    for (auto [lhsDim, rhsDim] :
         llvm::zip_equal(lhsBatchingDims, rhsBatchingDims)) {
      builder.addFactor(
          {lhsDim, rhsDim,
           mode != RaggedDotMode::kBatch ? groupSizesDim++ : kNullDim},
          outputDim++, lhsType.getDimSize(lhsDim));
    }

    // lhs non-contracting dimensions
    bool addLhsNonContractingDimsInGroupSizes =
        mode == RaggedDotMode::kNonContracting;
    for (int64_t i = 0; i < lhsType.getRank(); i++) {
      if (!llvm::is_contained(lhsContractingDims, i) &&
          !llvm::is_contained(lhsBatchingDims, i)) {
        FactorType factorType = FactorType::kPassThrough;
        if (i == lhsRaggedDim) {
          // We only add the non-contracting dimensions before the lhs ragged
          // dimension to the group sizes in the kNonContracting mode.
          addLhsNonContractingDimsInGroupSizes = false;
          factorType = FactorType::kNeedReplication;
        }
        builder.addFactor(
            {i, kNullDim,
             addLhsNonContractingDimsInGroupSizes ? groupSizesDim++ : kNullDim},
            outputDim++, lhsType.getDimSize(i), factorType);
      }
    }

    // rhs non-contracting dimensions
    for (int64_t i = 0; i < rhsType.getRank(); i++) {
      if (!llvm::is_contained(rhsContractingDims, i) &&
          !llvm::is_contained(rhsBatchingDims, i) &&
          !llvm::is_contained(rhsGroupDims, i)) {
        builder.addFactor({kNullDim, i, kNullDim}, outputDim++,
                          rhsType.getDimSize(i));
      }
    }

    // contracting dimensions
    bool addContractingDimsInGroupSizes = mode == RaggedDotMode::kContracting;
    for (auto [lhsDim, rhsDim] :
         llvm::zip_equal(lhsContractingDims, rhsContractingDims)) {
      FactorType factorType = FactorType::kReduction;
      if (lhsDim == lhsRaggedDim) {
        // We only add the contracting dimensions before the lhs ragged
        // dimension to the group sizes in the kContracting mode.
        addContractingDimsInGroupSizes = false;
        factorType = FactorType::kNeedReplication;
      }
      builder.addFactor(
          {lhsDim, rhsDim,
           addContractingDimsInGroupSizes ? groupSizesDim++ : kNullDim},
          kNullDim, lhsType.getDimSize(lhsDim), factorType);
    }

    switch (mode) {
      case RaggedDotMode::kNonContracting: {
        CHECK_EQ(groupSizesDim, groupSizesType.getRank() - 1);
        int64_t rhsGroupDim = rhsGroupDims.front();
        builder.addFactor({kNullDim, rhsGroupDim, groupSizesDim}, kNullDim,
                          rhsType.getDimSize(rhsGroupDim),
                          FactorType::kNeedReplication);
        break;
      }
      case RaggedDotMode::kContracting: {
        CHECK_EQ(groupSizesDim, groupSizesType.getRank() - 1);
        builder.addFactor({kNullDim, kNullDim, groupSizesDim}, 0,
                          groupSizesType.getDimSize(groupSizesDim),
                          FactorType::kNeedReplication);
        break;
      }
      case RaggedDotMode::kBatch: {
        break;
      }
    }

    return builder.build();
  }
};

}  // namespace

void registerMhloExtensions(mlir::DialectRegistry& registry) {
  registry.addExtension(+[](mlir::MLIRContext* ctx, mhlo::MhloDialect*) {
    mhlo::CopyOp::attachInterface<CopyShardingRuleOpInterface>(*ctx);
    mhlo::RaggedDotOp::attachInterface<RaggedDotShardingRuleOpInterface>(*ctx);
  });
}

}  // namespace sdy
}  // namespace xla
