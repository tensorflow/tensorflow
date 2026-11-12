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

#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"
#include "xla/hlo/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/utils/unregistered_attributes.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace stablehlo = ::mlir::stablehlo;

namespace xla {
namespace sdy {

bool isFullyManual(mlir::sdy::MeshAttr mesh,
                   mlir::ArrayRef<mlir::sdy::AxisRefAttr> manualAxes) {
  if (manualAxes.empty()) {
    return mesh.getAxes().empty();
  }
  if (mesh.getAxes().empty()) {
    return false;
  }
  mlir::SmallVector<mlir::sdy::AxisRefAttr> mergedManualAxes(manualAxes.begin(),
                                                             manualAxes.end());
  mlir::sdy::sortAndMergeAxes(mergedManualAxes, mesh);
  return mesh.getAxes().size() == mergedManualAxes.size() &&
         llvm::all_of(mergedManualAxes, [](mlir::sdy::AxisRefAttr axis) {
           return !axis.getSubAxisInfo();
         });
}

namespace {

using ::mlir::ArrayRef;
using ::mlir::DictionaryAttr;
using ::mlir::ModuleOp;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::OperationPass;
using ::mlir::Pass;
using ::mlir::PassWrapper;
using ::mlir::SmallVector;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::SymbolTable;
using ::mlir::func::FuncOp;

using ::mlir::sdy::AxisRefAttr;
using ::mlir::sdy::kShardingAttr;
using ::mlir::sdy::ManualAxesAttr;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::MeshOp;
using ::mlir::sdy::SdyDialect;
using ::mlir::sdy::TensorShardingAttr;

SmallVector<AxisRefAttr> getOrderedAxisRefs(TensorShardingAttr sdySharding,
                                            MeshAttr mesh,
                                            ArrayRef<AxisRefAttr> manualAxes) {
  if (llvm::none_of(manualAxes, [](AxisRefAttr axis) {
        return static_cast<bool>(axis.getSubAxisInfo());
      })) {
    return mlir::sdy::getOrderedAxisRefs(sdySharding, mesh);
  }
  SmallVector<AxisRefAttr> allAxes(manualAxes.begin(), manualAxes.end());
  for (mlir::sdy::DimensionShardingAttr dimSharding :
       sdySharding.getDimShardings()) {
    allAxes.append(dimSharding.getAxes().begin(), dimSharding.getAxes().end());
  }
  allAxes.append(sdySharding.getUnreducedAxes().begin(),
                 sdySharding.getUnreducedAxes().end());
  return mlir::sdy::getOrderedAxisRefs(
      mlir::sdy::AxisRefListAttr::get(sdySharding.getContext(), allAxes), mesh);
}

ArrayRef<AxisRefAttr> getValueManualAxes(mlir::Value v) {
  llvm::SmallPtrSet<mlir::Value, 8> visited;
  auto impl = [&](auto& self, mlir::Value val) -> ArrayRef<AxisRefAttr> {
    if (!val || !visited.insert(val).second) {
      return {};
    }
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(val)) {
      Operation* parentOp = blockArg.getOwner()->getParentOp();
      if (auto funcOp = mlir::dyn_cast_or_null<FuncOp>(parentOp);
          funcOp && blockArg.getOwner() == &funcOp.front()) {
        if (auto manualAxesAttr = funcOp.getArgAttrOfType<ManualAxesAttr>(
                blockArg.getArgNumber(), kManualAxes)) {
          return manualAxesAttr.getValue();
        }
        return {};
      }
      if (auto whileOp = mlir::dyn_cast_or_null<stablehlo::WhileOp>(parentOp)) {
        unsigned idx = blockArg.getArgNumber();
        if (idx < whileOp->getNumOperands()) {
          ArrayRef<AxisRefAttr> axes = self(self, whileOp->getOperand(idx));
          if (!axes.empty()) {
            return axes;
          }
        }
        Operation* bodyTerm = whileOp.getBody().front().getTerminator();
        if (idx < bodyTerm->getNumOperands()) {
          ArrayRef<AxisRefAttr> axes = self(self, bodyTerm->getOperand(idx));
          if (!axes.empty()) {
            return axes;
          }
        }
      }
      return {};
    }
    Operation* defOp = val.getDefiningOp();
    if (!defOp) {
      return {};
    }
    if (auto customCall = mlir::dyn_cast<stablehlo::CustomCallOp>(defOp)) {
      StringRef target = customCall.getCallTargetName();
      if (target == kSPMDShardToFullShapeCallTargetName ||
          target == kLocalToGlobalShapeCallTargetName) {
        return {};
      }
    }
    if (auto manualAxesAttr =
            defOp->getAttrOfType<ManualAxesAttr>(kManualAxes)) {
      return manualAxesAttr.getValue();
    }
    unsigned resIdx = mlir::cast<mlir::OpResult>(val).getResultNumber();
    if (auto whileOp = mlir::dyn_cast<stablehlo::WhileOp>(defOp)) {
      if (resIdx < whileOp->getNumOperands()) {
        ArrayRef<AxisRefAttr> axes = self(self, whileOp->getOperand(resIdx));
        if (!axes.empty()) {
          return axes;
        }
      }
      Operation* bodyTerm = whileOp.getBody().front().getTerminator();
      if (resIdx < bodyTerm->getNumOperands()) {
        ArrayRef<AxisRefAttr> axes = self(self, bodyTerm->getOperand(resIdx));
        if (!axes.empty()) {
          return axes;
        }
      }
      return {};
    }
    if (auto barrierOp =
            mlir::dyn_cast<stablehlo::OptimizationBarrierOp>(defOp)) {
      if (resIdx < barrierOp->getNumOperands()) {
        return self(self, barrierOp->getOperand(resIdx));
      }
      return {};
    }
    if (mlir::isa<mlir::mhlo::CopyOp, mlir::sdy::ReshardOp,
                  mlir::sdy::ShardingConstraintOp>(defOp) &&
        defOp->getNumOperands() == 1) {
      return self(self, defOp->getOperand(0));
    }
    return {};
  };
  return impl(impl, v);
}

// Convert the shardings from kShardingAttr into kXlaShardingAttr.
void exportFunc(FuncOp funcOp, const SymbolTable& symbolTable,
                OpBuilder& builder, bool addMissingShardingToControlFlow,
                bool enableHloShardingV3, bool simplifyReplicatedShardings,
                bool clearReverseOpSharding) {
  std::function<StringAttr(const HloSharding&)> getStringAttr =
      [&](const HloSharding& hloSharding) {
        return builder.getStringAttr(
            hloSharding.ToString(/*include_metadata=*/true));
      };
  std::function<MeshAttr(TensorShardingAttr)> getMeshAttr =
      [&](TensorShardingAttr sharding) {
        CHECK(sharding) << "null sharding while exporting func @"
                        << funcOp.getSymName().str();
        return sharding.getMesh(symbolTable);
      };

  // If clearReverseOpSharding is enabled, do a pre-pass walk specifically for
  // stablehlo.reverse operations before the function arguments layouts are
  // converted and erased. This ensures we can resolve block argument shardings.
  if (clearReverseOpSharding) {
    funcOp.front().walk([&](stablehlo::ReverseOp reverseOp) {
      if (mlir::sdy::getSharding(reverseOp.getOperand())) {
        reverseOp->removeAttr(kManualAxes);
        reverseOp->removeAttr(kShardingAttr);
      }
    });
  }

  if (addMissingShardingToControlFlow) {
    const HloSharding replicatedSharding =
        enableHloShardingV3 ? HloSharding(NamedSharding::Replicate())
                            : HloSharding::Replicate();
    const HloSharding singleDevice0 =
        enableHloShardingV3 ? HloSharding(NamedSharding::SingleDevice(0))
                            : HloSharding::SingleDevice(0);
    funcOp.front().walk([&](Operation* op) {
      if (!mlir::sdy::getShardings(op).empty() ||
          op->hasAttr(kXlaShardingAttr)) {
        return;
      }
      if (auto infeedOp = mlir::dyn_cast<stablehlo::InfeedOp>(op)) {
        if (infeedOp->getNumResults() == 1) {
          infeedOp->setAttr(kXlaShardingAttr, getStringAttr(singleDevice0));
          return;
        }
        std::vector<HloSharding> elemShardings;
        elemShardings.reserve(infeedOp->getNumResults());
        for (unsigned i = 0; i + 1 < infeedOp->getNumResults(); ++i) {
          mlir::Value val = infeedOp->getResult(i);
          TensorShardingAttr userSharding;
          ArrayRef<AxisRefAttr> userManualAxes;
          for (Operation* user : val.getUsers()) {
            if (auto castCall = mlir::dyn_cast<stablehlo::CustomCallOp>(user);
                castCall &&
                castCall.getCallTargetName() == "mhlo.bounded_dynamic_cast") {
              val = castCall.getResult(0);
              userSharding = mlir::sdy::getSharding(val);
              if (auto manualAttr =
                      castCall->getAttrOfType<ManualAxesAttr>(kManualAxes)) {
                userManualAxes = manualAttr.getValue();
              }
              break;
            }
          }
          if (!userSharding) {
            for (mlir::OpOperand& use : val.getUses()) {
              Operation* u = use.getOwner();
              if (mlir::isa<mlir::func::ReturnOp>(u)) {
                userSharding = funcOp.getResultAttrOfType<TensorShardingAttr>(
                    use.getOperandNumber(), kShardingAttr);
                if (auto manualAttr =
                        funcOp.getResultAttrOfType<ManualAxesAttr>(
                            use.getOperandNumber(), kManualAxes)) {
                  userManualAxes = manualAttr.getValue();
                }
              } else if (u->getNumResults() == 1 &&
                         u->getResult(0).getType() == val.getType()) {
                auto customCall = mlir::dyn_cast<stablehlo::CustomCallOp>(u);
                const bool isIdentityDimMapping =
                    u->hasTrait<mlir::OpTrait::Elementwise>() ||
                    mlir::isa<mlir::mhlo::CopyOp, mlir::sdy::ReshardOp,
                              mlir::sdy::ShardingConstraintOp>(u) ||
                    (customCall && customCall.getCallTargetName() ==
                                       kShardingCustomCallTargetName);
                if (!isIdentityDimMapping) {
                  continue;
                }
                userSharding = mlir::sdy::getSharding(u->getResult(0));
                userManualAxes = getValueManualAxes(u->getResult(0));
              }
              if (userSharding) {
                break;
              }
            }
          }
          if (userSharding) {
            elemShardings.push_back(convertToHloSharding(
                userSharding, getMeshAttr, userManualAxes, enableHloShardingV3,
                simplifyReplicatedShardings));
          } else {
            elemShardings.push_back(replicatedSharding);
          }
        }
        elemShardings.push_back(singleDevice0);
        std::vector<xla::Shape> shapes;
        shapes.reserve(infeedOp->getNumResults());
        llvm::transform(infeedOp->getResultTypes(), std::back_inserter(shapes),
                        [](mlir::Type t) { return xla::TypeToShape(t); });
        infeedOp->setAttr(
            kXlaShardingAttr,
            getStringAttr(HloSharding::Tuple(
                xla::ShapeUtil::MakeTupleShape(shapes), elemShardings)));
      } else if (mlir::isa<stablehlo::OutfeedOp, stablehlo::AfterAllOp>(op)) {
        op->setAttr(kXlaShardingAttr, getStringAttr(replicatedSharding));
      } else if (auto sendOp = mlir::dyn_cast<stablehlo::SendOp>(op);
                 sendOp && sendOp.getIsHostTransfer()) {
        sendOp->setAttr(kXlaShardingAttr, getStringAttr(singleDevice0));
      } else if (auto recvOp = mlir::dyn_cast<stablehlo::RecvOp>(op);
                 recvOp && recvOp.getIsHostTransfer()) {
        if (recvOp->getNumResults() == 1) {
          recvOp->setAttr(kXlaShardingAttr, getStringAttr(singleDevice0));
        } else {
          std::vector<HloSharding> elemShardings(recvOp->getNumResults(),
                                                 singleDevice0);
          std::vector<xla::Shape> shapes;
          shapes.reserve(recvOp->getNumResults());
          llvm::transform(recvOp->getResultTypes(), std::back_inserter(shapes),
                          [](mlir::Type t) { return xla::TypeToShape(t); });
          recvOp->setAttr(
              kXlaShardingAttr,
              getStringAttr(HloSharding::Tuple(
                  xla::ShapeUtil::MakeTupleShape(shapes), elemShardings)));
        }
      }
    });
  }

  llvm::SmallVector<mlir::DictionaryAttr> funcArgAttrs;
  funcArgAttrs.reserve(funcOp.getNumArguments());
  bool anyChanged = false;
  for (int64_t argNum = 0; argNum < funcOp.getNumArguments(); ++argNum) {
    mlir::NamedAttrList attrs(funcOp.getArgAttrDict(argNum));
    if (auto sdySharding = funcOp.getArgAttrOfType<TensorShardingAttr>(
            argNum, kShardingAttr)) {
      ArrayRef<AxisRefAttr> manualAxes;
      if (ManualAxesAttr manualAxesAttr =
              funcOp.getArgAttrOfType<ManualAxesAttr>(argNum, kManualAxes)) {
        manualAxes = manualAxesAttr.getValue();
        if (mlir::isa<stablehlo::TokenType>(
                funcOp.getArgument(argNum).getType()) &&
            !isFullyManual(getMeshAttr(sdySharding), manualAxes)) {
          manualAxes = {};
        }
      }
      attrs.set(kXlaShardingAttr,
                getStringAttr(convertToHloSharding(
                    sdySharding, getMeshAttr, manualAxes, enableHloShardingV3,
                    simplifyReplicatedShardings)));
      attrs.erase(kShardingAttr);
      anyChanged = true;
    }
    funcArgAttrs.push_back(attrs.getDictionary(funcOp.getContext()));
  }
  if (anyChanged) {
    funcOp.setAllArgAttrs(funcArgAttrs);
  }

  SmallVector<mlir::DictionaryAttr> newResultAttrs;
  newResultAttrs.reserve(funcOp.getNumResults());
  for (int64_t resNum = 0; resNum < funcOp.getNumResults(); ++resNum) {
    mlir::NamedAttrList attrs(funcOp.getResultAttrDict(resNum));
    if (auto sdySharding = mlir::dyn_cast_or_null<TensorShardingAttr>(
            attrs.get(kShardingAttr))) {
      ArrayRef<AxisRefAttr> manualAxes;
      if (auto manualAxesAttr =
              mlir::dyn_cast_or_null<ManualAxesAttr>(attrs.get(kManualAxes))) {
        manualAxes = manualAxesAttr.getValue();
        if (mlir::isa<stablehlo::TokenType>(funcOp.getResultTypes()[resNum]) &&
            !isFullyManual(getMeshAttr(sdySharding), manualAxes)) {
          manualAxes = {};
        }
        attrs.erase(kManualAxes);
      }
      attrs.set(kXlaShardingAttr,
                getStringAttr(convertToHloSharding(
                    sdySharding, getMeshAttr, manualAxes, enableHloShardingV3,
                    simplifyReplicatedShardings)));
      attrs.erase(kShardingAttr);
    }
    newResultAttrs.push_back(attrs.getDictionary(funcOp.getContext()));
  }
  funcOp.setAllResultAttrs(newResultAttrs);

  funcOp.front().walk([&](Operation* op) {
    ArrayRef<AxisRefAttr> manualAxes;
    if (ManualAxesAttr manualAxesAttr =
            op->getAttrOfType<ManualAxesAttr>(kManualAxes)) {
      manualAxes = manualAxesAttr.getValue();
    }

    if (ArrayRef<TensorShardingAttr> shardings = mlir::sdy::getShardings(op);
        !shardings.empty()) {
      setHloShardingAttr(op, shardings, getMeshAttr, manualAxes,
                         enableHloShardingV3, simplifyReplicatedShardings);
      op->removeAttr(kShardingAttr);
    } else if (addMissingShardingToControlFlow &&
               mlir::isa<stablehlo::WhileOp, stablehlo::CaseOp,
                         stablehlo::IfOp>(op)) {
      // The shard map export pass assigns shardings to any operation with
      // manual axes. Since this operation lacks shardings
      // (`shardings.empty()`), it cannot have manual axes. This CHECK asserts
      // this invariant before we add a replicated sharding.
      CHECK(manualAxes.empty());
      const HloSharding replicatedSharding =
          enableHloShardingV3 ? HloSharding(NamedSharding::Replicate())
                              : HloSharding::Replicate();
      op->setAttr(kXlaShardingAttr, getStringAttr(replicatedSharding));
    }
  });

  for (int64_t argNum = 0; argNum < funcOp.getNumArguments(); ++argNum) {
    funcOp.removeArgAttr(argNum, kManualAxes);
  }
  funcOp.front().walk([&](Operation* op) { op->removeAttr(kManualAxes); });
}

void exportModuleShardingMetadata(ModuleOp moduleOp,
                                  const SymbolTable& symbolTable,
                                  OpBuilder& builder, bool enableHloShardingV3,
                                  bool simplifyReplicatedShardings) {
  auto getStringAttr = [&](const HloSharding& hloSharding) {
    return builder.getStringAttr(
        hloSharding.ToString(/*include_metadata=*/true));
  };
  auto toHloSharding = [&](TensorShardingAttr sdySharding) {
    return convertToHloSharding(
        sdySharding,
        [&](TensorShardingAttr sharding) {
          return sharding.getMesh(symbolTable);
        },
        /*manualAxes=*/{}, enableHloShardingV3, simplifyReplicatedShardings);
  };

  if (auto paramShardingsAttr =
          moduleOp->getAttrOfType<mlir::sdy::TensorShardingPerValueAttr>(
              mlir::sdy::kParametersShardingsAttr)) {
    SmallVector<mlir::Attribute> paramShardingAttrs;
    paramShardingAttrs.reserve(paramShardingsAttr.size());
    for (TensorShardingAttr sdySharding : paramShardingsAttr.getShardings()) {
      paramShardingAttrs.push_back(getStringAttr(toHloSharding(sdySharding)));
    }
    moduleOp->setAttr(xla::kMhloSpmdParametersShardings,
                      builder.getArrayAttr(paramShardingAttrs));
    moduleOp->removeAttr(mlir::sdy::kParametersShardingsAttr);
  }

  if (auto outputShardingsAttr =
          moduleOp->getAttrOfType<mlir::sdy::TensorShardingPerValueAttr>(
              mlir::sdy::kOutputShardingsAttr)) {
    ArrayRef<TensorShardingAttr> sdyShardings =
        outputShardingsAttr.getShardings();
    HloSharding outputSharding = [&]() {
      if (sdyShardings.size() == 1) {
        return toHloSharding(sdyShardings.front());
      }
      std::vector<HloSharding> outputHloShardings;
      outputHloShardings.reserve(sdyShardings.size());
      llvm::transform(sdyShardings, std::back_inserter(outputHloShardings),
                      toHloSharding);
      return HloSharding::FlatTuple(std::move(outputHloShardings));
    }();
    moduleOp->setAttr(xla::kMhloSpmdOutputSharding,
                      getStringAttr(outputSharding));
    moduleOp->removeAttr(mlir::sdy::kOutputShardingsAttr);
  }
}

class ExportStablehloShardingsPass
    : public PassWrapper<ExportStablehloShardingsPass,
                         OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExportStablehloShardingsPass)

  explicit ExportStablehloShardingsPass(
      bool addMissingShardingToControlFlow, bool enableHloShardingV3 = false,
      bool simplifyReplicatedShardings = false,
      bool clearReverseOpSharding = false)
      : addMissingShardingToControlFlow(addMissingShardingToControlFlow),
        enableHloShardingV3(enableHloShardingV3),
        simplifyReplicatedShardings(simplifyReplicatedShardings),
        clearReverseOpSharding(clearReverseOpSharding) {}

  StringRef getArgument() const override {
    return "xla-sdy-stablehlo-export-shardings";
  }

  StringRef getDescription() const override {
    return "Converts the shardings from kShardingAttr to kXlaShardingAttr and "
           "removes mesh symbols.";
  }

  void getDependentDialects(mlir::DialectRegistry& registry) const final {
    registry.insert<SdyDialect>();
  }

 protected:
  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    mlir::SymbolTableCollection symbolTableCollection;
    SymbolTable& symbolTable = symbolTableCollection.getSymbolTable(moduleOp);

    auto builder = OpBuilder::atBlockBegin(&moduleOp.getBodyRegion().front());

    exportModuleShardingMetadata(moduleOp, symbolTable, builder,
                                 enableHloShardingV3,
                                 simplifyReplicatedShardings);

    for (auto funcOp : moduleOp.getOps<FuncOp>()) {
      exportFunc(funcOp, symbolTable, builder, addMissingShardingToControlFlow,
                 enableHloShardingV3, simplifyReplicatedShardings,
                 clearReverseOpSharding);
    }

    // Collect all used mesh names from StableHLO attributes
    llvm::DenseSet<StringRef> usedMeshNames;
    auto collectMeshesFromReplicaGroups = [&](Operation* op) {
      if (auto attr = op->getAttr("replica_groups")) {
        if (auto rgv3 =
                mlir::dyn_cast<stablehlo::ReplicaGroupMeshAxesAttr>(attr)) {
          mlir::Attribute meshAttr = rgv3.getMesh();
          if (auto symbolRef = mlir::dyn_cast<mlir::SymbolRefAttr>(meshAttr)) {
            usedMeshNames.insert(symbolRef.getLeafReference().getValue());
          } else if (auto stringAttr =
                         mlir::dyn_cast<mlir::StringAttr>(meshAttr)) {
            usedMeshNames.insert(stringAttr.getValue());
          }
        }
      }
    };

    moduleOp.walk([&](Operation* op) {
      if (auto customCall = mlir::dyn_cast<stablehlo::CustomCallOp>(op)) {
        // StableHLO doesn't have an equivalent of `erf` and `topk` ops.
        // If they have a sharding annotation, we need to move it into
        // `mhlo.attributes`, which StableHLO->MHLO conversion would lift back
        // up.
        StringRef callTargetName = customCall.getCallTargetName();
        if (callTargetName == "mhlo.erf" || callTargetName == "mhlo.topk") {
          // TODO(bartchr): refactor `addFrontendAttribute` to take a key for
          // the dictionary attribute. Then can re-use the logic instead of
          // duplicating it here for `kMhloAttributesAttr`.
          if (auto sdySharding =
                  customCall->getAttrOfType<StringAttr>(kXlaShardingAttr)) {
            customCall->removeAttr(kXlaShardingAttr);
            SmallVector<mlir::NamedAttribute> newAttributes(
                customCall->getAttrOfType<DictionaryAttr>(kMhloAttributesAttr)
                    .getValue());
            newAttributes.push_back(
                builder.getNamedAttr(kXlaShardingAttr, sdySharding));
            customCall->setAttr(kMhloAttributesAttr,
                                builder.getDictionaryAttr(newAttributes));
          }
        }
      } else if (mlir::isa<stablehlo::AllGatherOp, stablehlo::AllReduceOp,
                           stablehlo::ReduceScatterOp, stablehlo::AllToAllOp,
                           stablehlo::CollectiveBroadcastOp>(op)) {
        collectMeshesFromReplicaGroups(op);
      }
    });

    // Remove all mesh symbols that are not used
    for (MeshOp meshOp :
         llvm::make_early_inc_range(moduleOp.getOps<MeshOp>())) {
      if (!usedMeshNames.contains(meshOp.getName())) {
        symbolTable.erase(meshOp);
      }
    }
  }

 private:
  bool addMissingShardingToControlFlow;
  bool enableHloShardingV3;
  bool simplifyReplicatedShardings;
  bool clearReverseOpSharding;
};

HloSharding getHloShardingForOp(
    Operation* op, ArrayRef<TensorShardingAttr> shardings,
    std::function<MeshAttr(TensorShardingAttr)> getMeshAttr,
    ArrayRef<AxisRefAttr> manualAxes, bool enableHloShardingV3,
    bool simplifyReplicatedShardings) {
  auto getManualAxesForResult = [&](unsigned resIdx, mlir::Type type,
                                    TensorShardingAttr sdySharding) {
    ArrayRef<AxisRefAttr> resManualAxes = manualAxes;
    if (resManualAxes.empty() && resIdx < op->getNumResults() &&
        mlir::isa<stablehlo::WhileOp, stablehlo::OptimizationBarrierOp,
                  mlir::mhlo::CopyOp, mlir::sdy::ReshardOp,
                  mlir::sdy::ShardingConstraintOp>(op)) {
      resManualAxes = getValueManualAxes(op->getResult(resIdx));
    }
    if (mlir::isa<stablehlo::TokenType>(type) &&
        !isFullyManual(getMeshAttr(sdySharding), resManualAxes)) {
      return ArrayRef<AxisRefAttr>();
    }
    return resManualAxes;
  };

  bool isNoResultSingleDevice =
      op->getNumResults() == 0 && shardings.size() == 1 &&
      (getMeshAttr(shardings.front()).isSingleDevice() ||
       shardings.front().isFullyReplicated());
  CHECK(shardings.size() == op->getNumResults() || isNoResultSingleDevice);
  if (op->getNumResults() == 1 || isNoResultSingleDevice) {
    ArrayRef<AxisRefAttr> resultManualAxes =
        op->getNumResults() == 1
            ? getManualAxesForResult(0, op->getResultTypes().front(),
                                     shardings.front())
            : manualAxes;
    return convertToHloSharding(shardings.front(), getMeshAttr,
                                resultManualAxes, enableHloShardingV3,
                                simplifyReplicatedShardings);
  }

  std::vector<HloSharding> newShardings;
  newShardings.reserve(shardings.size());
  for (auto [resIdx, type, sdySharding] :
       llvm::enumerate(op->getResultTypes(), shardings)) {
    newShardings.push_back(
        convertToHloSharding(sdySharding, getMeshAttr,
                             getManualAxesForResult(resIdx, type, sdySharding),
                             enableHloShardingV3, simplifyReplicatedShardings));
  }

  std::vector<xla::Shape> shapes;
  shapes.reserve(op->getNumResults());
  llvm::transform(op->getResultTypes(), std::back_inserter(shapes),
                  [&](mlir::Type type) { return xla::TypeToShape(type); });

  return HloSharding::Tuple(xla::ShapeUtil::MakeTupleShape(shapes),
                            newShardings);
}

// Converts an sdy::AxisRefAttr to an xla::AxisRef.
AxisRef toAxisRef(AxisRefAttr axisRefAttr,
                  const llvm::DenseMap<StringRef, int64_t>& axisNameToIndex) {
  auto it = axisNameToIndex.find(axisRefAttr.getName());
  CHECK(it != axisNameToIndex.end());
  int64_t meshAxisIndex = it->getSecond();
  if (mlir::sdy::SubAxisInfoAttr subAxisInfo = axisRefAttr.getSubAxisInfo()) {
    return AxisRef(meshAxisIndex,
                   {subAxisInfo.getPreSize(), subAxisInfo.getSize()});
  }
  return AxisRef(meshAxisIndex);
}

NamedSharding convertToNamedSharding(TensorShardingAttr sdySharding,
                                     MeshAttr sdyMesh,
                                     ArrayRef<AxisRefAttr> manualAxes,
                                     bool simplifyReplicatedShardings) {
  // If simplifyReplicatedShardings is enabled, we convert fully replicated,
  // closed, non-manual, and non-unreduced shardings to
  // NamedSharding::Replicate(), which is represented as "{mesh[], replicated}".
  // We do this only during the final export at the end of ShardyXLA pass to
  // make the final HLO strings much shorter and more human-readable. We must
  // NOT do this during the initial MLIR-to-HLO export, because preserving the
  // rank and open/closed dimension states is required for Shardy verification
  // passes.
  if (simplifyReplicatedShardings && sdySharding.isFullyReplicated() &&
      sdySharding.isFullyClosed() && sdySharding.getUnreducedAxes().empty() &&
      manualAxes.empty() && sdyMesh.getDeviceIds().empty()) {
    return NamedSharding::Replicate();
  }
  // If there are no axes, convert to:
  // - single-device sharding if the mesh has a device id
  // - else replicated sharding
  if (sdyMesh.getAxes().empty()) {
    if (!sdyMesh.getDeviceIds().empty()) {
      return NamedSharding::SingleDevice(sdyMesh.getDeviceIds().front());
    }

    SmallVector<NamedSharding::DimensionSharding> dimShardings;
    dimShardings.reserve(sdySharding.getDimShardings().size());
    for (mlir::sdy::DimensionShardingAttr dimSharding :
         sdySharding.getDimShardings()) {
      dimShardings.push_back(
          NamedSharding::DimensionSharding({}, dimSharding.getIsClosed()));
    }
    return NamedSharding(Mesh(), dimShardings);
  }

  std::vector<int64_t> axisSizes;
  std::vector<absl::string_view> axisNames;
  llvm::DenseMap<StringRef, int64_t> axisNameToIndex;
  for (auto [index, axis] : llvm::enumerate(sdyMesh.getAxes())) {
    axisSizes.push_back(axis.getSize());
    axisNames.emplace_back(axis.getName().data(), axis.getName().size());
    axisNameToIndex[axis.getName()] = index;
  }

  Mesh mesh;
  if (sdyMesh.getDeviceIds().empty()) {
    mesh = Mesh(axisSizes, axisNames);
  } else {
    LOG(WARNING) << "This branch is not expected as JAX is not known to use "
                    "device lists.";
    Array<int64_t> deviceIdsArray(axisSizes, sdyMesh.getDeviceIds());
    mesh = Mesh(deviceIdsArray, axisNames);
  }

  if (isFullyManual(sdyMesh, manualAxes)) {
    // Every axis is manual, so there is nothing left to shard over. HLO's
    // canonical form for this omits the dimension shardings entirely; import
    // restores them from the value's type.
    return NamedSharding::Manual(mesh);
  }

  SmallVector<NamedSharding::DimensionSharding> dimShardings;
  dimShardings.reserve(sdySharding.getDimShardings().size());
  for (mlir::sdy::DimensionShardingAttr dimSharding :
       sdySharding.getDimShardings()) {
    SmallVector<AxisRef> axes;
    axes.reserve(dimSharding.getAxes().size());
    for (AxisRefAttr axisRefAttr : dimSharding.getAxes()) {
      axes.push_back(toAxisRef(axisRefAttr, axisNameToIndex));
    }
    dimShardings.push_back(
        NamedSharding::DimensionSharding(axes, dimSharding.getIsClosed()));
  }

  SmallVector<AxisRef> replicatedAxes;
  replicatedAxes.reserve(sdySharding.getReplicatedAxes().size());
  for (AxisRefAttr axisRefAttr : sdySharding.getReplicatedAxes()) {
    replicatedAxes.push_back(toAxisRef(axisRefAttr, axisNameToIndex));
  }

  SmallVector<AxisRef> unreducedAxes;
  unreducedAxes.reserve(sdySharding.getUnreducedAxes().size());
  for (AxisRefAttr axisRefAttr : sdySharding.getUnreducedAxes()) {
    unreducedAxes.push_back(toAxisRef(axisRefAttr, axisNameToIndex));
  }

  SmallVector<AxisRef> manualAxesSharding;
  manualAxesSharding.reserve(manualAxes.size());
  for (AxisRefAttr axisRefAttr : manualAxes) {
    manualAxesSharding.push_back(toAxisRef(axisRefAttr, axisNameToIndex));
  }

  ReductionOp reductionOp;
  switch (sdySharding.getReductionOp()) {
    case mlir::sdy::ReductionOp::SUM:
      reductionOp = ReductionOp::kSum;
      break;
    case mlir::sdy::ReductionOp::MAX:
      reductionOp = ReductionOp::kMax;
      break;
    case mlir::sdy::ReductionOp::MIN:
      reductionOp = ReductionOp::kMin;
      break;
  }

  return NamedSharding(mesh, dimShardings, replicatedAxes, unreducedAxes,
                       manualAxesSharding, /*metadata=*/{}, reductionOp);
}

}  // namespace

HloSharding convertToHloSharding(
    TensorShardingAttr sdySharding,
    std::function<MeshAttr(TensorShardingAttr)> getMeshAttr,
    ArrayRef<AxisRefAttr> manualAxes, bool enableHloShardingV3,
    bool simplifyReplicatedShardings) {
  MeshAttr mesh = getMeshAttr(sdySharding);
  SmallVector<AxisRefAttr> sortedManualAxes(manualAxes.begin(),
                                            manualAxes.end());
  if (!sortedManualAxes.empty() && !mesh.getAxes().empty()) {
    mlir::sdy::sortAndMergeAxes(sortedManualAxes, mesh);
    if (!isFullyManual(mesh, sortedManualAxes)) {
      bool hasOverlap = false;
      for (mlir::sdy::DimensionShardingAttr dimSharding :
           sdySharding.getDimShardings()) {
        if (mlir::sdy::overlaps(dimSharding.getAxes(), sortedManualAxes)) {
          hasOverlap = true;
          break;
        }
      }
      if (!hasOverlap && (mlir::sdy::overlaps(sdySharding.getReplicatedAxes(),
                                              sortedManualAxes) ||
                          mlir::sdy::overlaps(sdySharding.getUnreducedAxes(),
                                              sortedManualAxes))) {
        hasOverlap = true;
      }
      if (hasOverlap) {
        mlir::MLIRContext* ctx = sdySharding.getContext();
        SmallVector<mlir::sdy::DimensionShardingAttr> cleanDims;
        cleanDims.reserve(sdySharding.getRank());
        for (mlir::sdy::DimensionShardingAttr dimSharding :
             sdySharding.getDimShardings()) {
          SmallVector<AxisRefAttr> axes(dimSharding.getAxes().begin(),
                                        dimSharding.getAxes().end());
          mlir::sdy::truncateAxesByRemovingOverlaps(axes, sortedManualAxes);
          cleanDims.push_back(mlir::sdy::DimensionShardingAttr::get(
              ctx, axes, dimSharding.getIsClosed(), dimSharding.getPriority()));
        }
        SmallVector<AxisRefAttr> cleanReplicated(
            sdySharding.getReplicatedAxes().begin(),
            sdySharding.getReplicatedAxes().end());
        mlir::sdy::truncateAxesByRemovingOverlaps(cleanReplicated,
                                                  sortedManualAxes);
        SmallVector<AxisRefAttr> cleanUnreduced(
            sdySharding.getUnreducedAxes().begin(),
            sdySharding.getUnreducedAxes().end());
        mlir::sdy::truncateAxesByRemovingOverlaps(cleanUnreduced,
                                                  sortedManualAxes);
        sdySharding = TensorShardingAttr::get(
            ctx, sdySharding.getMeshOrRef(), cleanDims, cleanReplicated,
            cleanUnreduced, sdySharding.getReductionOp());
      }
    }
  }
  if (enableHloShardingV3) {
    return HloSharding(convertToNamedSharding(
        sdySharding, mesh, sortedManualAxes, simplifyReplicatedShardings));
  }

  // If there are no axes, convert to:
  // - single-device sharding if the mesh has a device id
  // - else replicated sharding
  if (mesh.getAxes().empty()) {
    return mesh.getDeviceIds().empty()
               ? HloSharding::Replicate()
               : HloSharding::SingleDevice(mesh.getDeviceIds().front());
  }

  SmallVector<int64_t> tileAssignmentDims(sdySharding.getRank(), 1);
  llvm::SmallDenseMap<AxisRefAttr, int64_t> axisRefToShardedPos;
  SmallVector<OpSharding::Type> types;
  int64_t shardedPos = 0;

  if (isFullyManual(mesh, sortedManualAxes)) {
    return HloSharding::Manual();
  }

  // Iterate the dim shardings.
  for (auto [index, dimSharding] :
       llvm::enumerate(sdySharding.getDimShardings())) {
    for (AxisRefAttr axisRef : dimSharding.getAxes()) {
      tileAssignmentDims[index] *= axisRef.getSize(mesh);
      axisRefToShardedPos[axisRef] = shardedPos++;
    }
  }

  // Iterate the manual axes.
  if (!sortedManualAxes.empty()) {
    types.push_back(OpSharding::MANUAL);
    int64_t& manualDim = tileAssignmentDims.emplace_back(1);
    for (AxisRefAttr manualAxis : sortedManualAxes) {
      manualDim *= manualAxis.getSize(mesh);
      axisRefToShardedPos[manualAxis] = shardedPos++;
    }
  }

  // Iterate the unreduced axes.
  if (!sdySharding.getUnreducedAxes().empty()) {
    types.push_back(OpSharding::UNREDUCED);
    int64_t& unreducedDim = tileAssignmentDims.emplace_back(1);
    for (AxisRefAttr unreducedAxis : sdySharding.getUnreducedAxes()) {
      unreducedDim *= unreducedAxis.getSize(mesh);
      axisRefToShardedPos[unreducedAxis] = shardedPos++;
    }
  }

  // We will add all axes and let canonicalization merge adjacent axes.
  SmallVector<AxisRefAttr> meshAxisRefs =
      getOrderedAxisRefs(sdySharding, mesh, sortedManualAxes);
  SmallVector<int64_t> reshapeDims(meshAxisRefs.size());
  SmallVector<int> transposePerm(meshAxisRefs.size());

  int64_t totalReplicatedSize = 1;
  int64_t replicatedPos = shardedPos;
  for (auto [axisIndex, axisRef] : llvm::enumerate(meshAxisRefs)) {
    reshapeDims[axisIndex] = axisRef.getSize(mesh);

    auto shardedPosIt = axisRefToShardedPos.find(axisRef);
    if (shardedPosIt == axisRefToShardedPos.end()) {
      // Axis is replicated.
      transposePerm[replicatedPos++] = axisIndex;
      totalReplicatedSize *= axisRef.getSize(mesh);
    } else {
      // Axis is sharded, manual, or unreduced.
      transposePerm[shardedPosIt->second] = axisIndex;
    }
  }

  if (totalReplicatedSize > 1) {
    tileAssignmentDims.push_back(totalReplicatedSize);
    types.push_back(OpSharding::REPLICATED);
  }

  ReductionOp reductionOp = ReductionOp::kSum;
  switch (sdySharding.getReductionOp()) {
    case mlir::sdy::ReductionOp::SUM:
      reductionOp = ReductionOp::kSum;
      break;
    case mlir::sdy::ReductionOp::MAX:
      reductionOp = ReductionOp::kMax;
      break;
    case mlir::sdy::ReductionOp::MIN:
      reductionOp = ReductionOp::kMin;
      break;
  }

  // Handle arbitrary device ID list.
  if (!mesh.getDeviceIds().empty()) {
    Array<int64_t> deviceIdsArray(reshapeDims);
    deviceIdsArray.SetValues(mesh.getDeviceIds());
    deviceIdsArray.TransposeDimensions(transposePerm);
    deviceIdsArray.Reshape(tileAssignmentDims);
    HloSharding res = HloSharding::Subgroup(
        TileAssignment(
            std::make_shared<const Array<int64_t>>(std::move(deviceIdsArray))),
        types);
    res.set_reduction_op(reductionOp);
    return res;
  }

  HloSharding res = HloSharding::Subgroup(
      xla::TileAssignment(tileAssignmentDims, reshapeDims, transposePerm),
      types);
  res.set_reduction_op(reductionOp);
  return res;
}

void setHloShardingAttr(Operation* op, ArrayRef<TensorShardingAttr> shardings,
                        std::function<MeshAttr(TensorShardingAttr)> getMeshAttr,
                        ArrayRef<AxisRefAttr> manualAxes,
                        bool enableHloShardingV3,
                        bool simplifyReplicatedShardings) {
  HloSharding hloSharding =
      getHloShardingForOp(op, shardings, getMeshAttr, manualAxes,
                          enableHloShardingV3, simplifyReplicatedShardings);
  op->setAttr(kXlaShardingAttr,
              StringAttr::get(op->getContext(),
                              hloSharding.ToString(/*include_metadata=*/true)));
}

std::unique_ptr<Pass> createExportStablehloShardingsPass(
    bool addMissingShardingToControlFlow, bool enableHloShardingV3,
    bool simplifyReplicatedShardings, bool clearReverseOpSharding) {
  return std::make_unique<ExportStablehloShardingsPass>(
      addMissingShardingToControlFlow, enableHloShardingV3,
      simplifyReplicatedShardings, clearReverseOpSharding);
}

void registerStablehloExportShardingsPass() {
  mlir::registerPass([]() {
    return createExportStablehloShardingsPass(
        /*addMissingShardingToControlFlow=*/false,
        /*enableHloShardingV3=*/false);
  });
}

}  // namespace sdy
}  // namespace xla
