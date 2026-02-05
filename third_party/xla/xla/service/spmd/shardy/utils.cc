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

#include "xla/service/spmd/shardy/utils.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "mhlo/IR/register.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/service/spmd/shardy/constants.h"
#include "xla/service/spmd/shardy/extensions/mhlo_extensions.h"

namespace xla {
namespace sdy {

using ::mlir::ArrayRef;
using ::mlir::Attribute;
using ::mlir::DictionaryAttr;
using ::mlir::NamedAttribute;
using ::mlir::Operation;
using ::mlir::SmallVector;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using xla::sdy::kFrontendAttributesAttr;

using ::mlir::func::FuncOp;
using ::mlir::sdy::AxisRefAttr;
using ::mlir::sdy::AxisRefListAttr;
using ::mlir::sdy::DimensionShardingAttr;
using ::mlir::sdy::MeshAttr;
using ::mlir::sdy::MeshAxisAttr;
using ::mlir::sdy::SubAxisInfoAttr;
using ::mlir::sdy::TensorShardingAttr;
using ::mlir::sdy::TensorShardingPerValueAttr;
using ::mlir::stablehlo::CustomCallOp;

absl::string_view toStringView(mlir::StringRef sr) {
  return absl::string_view(sr.data(), sr.size());
}

DictionaryAttr getFrontendAttrs(Operation* op) {
  return op->getAttrOfType<DictionaryAttr>(kFrontendAttributesAttr);
}

DictionaryAttr getFuncArgFrontendAttrs(FuncOp funcOp, unsigned int index) {
  return funcOp.getArgAttrOfType<DictionaryAttr>(index,
                                                 kFrontendAttributesAttr);
}

namespace {

mlir::StringAttr getStringAttribute(Attribute attr, mlir::OpBuilder& builder) {
  std::string value;
  if (auto stringAttr = mlir::dyn_cast<StringAttr>(attr)) {
    return stringAttr;
  }
  return builder.getStringAttr(mlir::sdy::attributeToString(attr));
}

SmallVector<NamedAttribute> getExistingFrontendAttributes(
    DictionaryAttr frontendAttributes, StringRef excludedAttribute) {
  SmallVector<NamedAttribute> dictEntries;
  if (!frontendAttributes) {
    return dictEntries;
  }
  for (NamedAttribute entry : frontendAttributes) {
    if (entry.getName() != excludedAttribute) {
      dictEntries.push_back(entry);
    }
  }
  return dictEntries;
}

void setFrontendAttribute(SmallVector<NamedAttribute>& existingAttributes,
                          StringRef name, Attribute value) {
  mlir::OpBuilder builder(value.getContext());
  StringAttr stringValue = getStringAttribute(value, builder);
  for (auto* it = existingAttributes.begin(); it != existingAttributes.end();
       ++it) {
    if (it->getName() == name) {
      if (it->getValue() == stringValue) {
        return;
      }
      existingAttributes.erase(it);
      break;
    }
  }
  existingAttributes.emplace_back(
      NamedAttribute(builder.getStringAttr(name), stringValue));
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

std::optional<TensorShardingAttr> adjustShardingInternal(
    mlir::MLIRContext* context, int idx, TensorShardingAttr sharding,
    int64_t rank, absl::Span<const bool> allowSpmdShardingPropagation) {
  bool allowPropagation = false;
  if (!allowSpmdShardingPropagation.empty()) {
    allowPropagation = allowSpmdShardingPropagation.size() == 1
                           ? allowSpmdShardingPropagation[0]
                           : allowSpmdShardingPropagation[idx];
  }

  if (allowPropagation) {
    return std::nullopt;
  }

  // Close all dimensions if sharding propagation is not allowed.
  if (sharding) {
    sharding = sharding.getClosedLike(sharding);
  } else {
    sharding = TensorShardingAttr::getFullyClosed(context, rank,
                                                  MeshAttr::get(context, {}));
  }

  return sharding;
}

}  // namespace

void setFrontendAttribute(Operation* op, StringRef name, Attribute value) {
  SmallVector<NamedAttribute> existingAttributes =
      getExistingFrontendAttributes(getFrontendAttrs(op), "");
  setFrontendAttribute(existingAttributes, name, value);
  setFrontendAttrs(op, existingAttributes);
}

void setFrontendAttribute(FuncOp funcOp, StringRef name, Attribute value,
                          int64_t argNum) {
  SmallVector<NamedAttribute> existingAttributes =
      getExistingFrontendAttributes(getFuncArgFrontendAttrs(funcOp, argNum),
                                    "");
  setFrontendAttribute(existingAttributes, name, value);
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

bool hasFrontendAttr(mlir::Operation* op, mlir::StringRef key) {
  return hasKey(getFrontendAttrs(op), key);
}

bool hasKey(mlir::DictionaryAttr dictAttr, mlir::StringRef key) {
  return dictAttr && dictAttr.contains(key);
}

void loadAllRequiredDialects(mlir::MLIRContext* context) {
  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  registerMhloExtensions(registry);
  mlir::sdy::registerAllDialects(registry);
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
}

void adjustInputSharding(
    FuncOp func, int idx, TensorShardingAttr sharding, int64_t rank,
    absl::Span<const bool> allowSpmdShardingPropagationToParameters) {
  if (std::optional<TensorShardingAttr> adjustedSharding =
          adjustShardingInternal(func.getContext(), idx, sharding, rank,
                                 allowSpmdShardingPropagationToParameters)) {
    mlir::sdy::setSharding(func.getArgument(idx), *adjustedSharding);
  }
}

void adjustOutputSharding(
    FuncOp func, int idx, TensorShardingAttr sharding, int64_t rank,
    absl::Span<const bool> allowSpmdShardingPropagationToOutput) {
  if (std::optional<TensorShardingAttr> adjustedSharding =
          adjustShardingInternal(func.getContext(), idx, sharding, rank,
                                 allowSpmdShardingPropagationToOutput)) {
    setFuncResultSharding(func, idx, *adjustedSharding);
  }
}

CustomCallOp cloneCustomCallWithNewResultTypes(CustomCallOp op,
                                               mlir::TypeRange resultTypes,
                                               mlir::IRRewriter& rewriter) {
  auto customCallOp = CustomCallOp::create(
      rewriter, op.getLoc(), resultTypes, op.getOperands(),
      op.getCallTargetNameAttr(), op.getHasSideEffectAttr(),
      op.getBackendConfigAttr(), op.getApiVersionAttr(),
      op.getCalledComputations(), op.getOperandLayoutsAttr(),
      op.getResultLayoutsAttr(), op.getOutputOperandAliases());
  customCallOp->setDiscardableAttrs(mlir::DictionaryAttr::get(
      op->getContext(), llvm::to_vector(op->getDiscardableAttrs())));
  return customCallOp;
};

bool isPythonCallbackCustomCall(mlir::stablehlo::CustomCallOp op) {
  mlir::StringRef targetName = op.getCallTargetName();
  return targetName == kPythonCpuCallbackCustomCallTargetName ||
         targetName == kPythonGpuCallbackCustomCallTargetName ||
         targetName == kFFIPythonCpuCallbackCustomCallTargetName ||
         targetName == kFFIPythonGpuCallbackCustomCallTargetName;
}

std::string duplicateShardingsAtIndices(
    mlir::StringRef shardingsFrontendAttr,
    const llvm::BitVector& indicesToDuplicate) {
  auto context = std::make_unique<mlir::MLIRContext>(
      mlir::MLIRContext::Threading::DISABLED);
  context->loadDialect<mlir::sdy::SdyDialect>();
  auto shardingPerValue = parseStringAttr<TensorShardingPerValueAttr>(
      shardingsFrontendAttr, context.get());
  CHECK(shardingPerValue);
  SmallVector<TensorShardingAttr> newShardings;
  newShardings.reserve(shardingPerValue.size());
  for (auto [index, sharding] :
       llvm::enumerate(shardingPerValue.getShardings())) {
    newShardings.push_back(sharding);
    if (indicesToDuplicate.test(index)) {
      newShardings.push_back(sharding);
    }
  }
  return mlir::sdy::attributeToString(
      TensorShardingPerValueAttr::get(context.get(), newShardings));
}

SmallVector<AxisRefAttr> getOrderedAxisRefs(Attribute shardingOrAxisList,
                                            MeshAttr mesh) {
  // We use a map vector to maintain the order of mesh axes.
  llvm::MapVector<StringRef, SmallVector<int64_t>> axisNameToPreSizes;
  axisNameToPreSizes.reserve(mesh.getAxes().size());
  for (MeshAxisAttr meshAxis : mesh.getAxes()) {
    SmallVector<int64_t>& preSizes = axisNameToPreSizes[meshAxis.getName()];
    preSizes.push_back(1);
    preSizes.push_back(meshAxis.getSize());
  }

  auto consumeAxisRefList = [&](ArrayRef<AxisRefAttr> axisRefs) {
    for (AxisRefAttr axisRef : axisRefs) {
      // Add sub-axis pre-sizes to `axisNameToPreSizes`. We'll dedup later.
      if (axisRef.getSubAxisInfo()) {
        SmallVector<int64_t>& preSizes = axisNameToPreSizes[axisRef.getName()];
        preSizes.push_back(axisRef.getSubAxisInfo().getPreSize());
        preSizes.push_back(axisRef.getSubAxisInfo().getNextPreSize());
      }
    }
  };

  if (auto sharding = mlir::dyn_cast<TensorShardingAttr>(shardingOrAxisList)) {
    for (DimensionShardingAttr dimSharding : sharding.getDimShardings()) {
      consumeAxisRefList(dimSharding.getAxes());
    }
    consumeAxisRefList(sharding.getUnreducedAxes());
  } else {
    consumeAxisRefList(
        mlir::cast<AxisRefListAttr>(shardingOrAxisList).getValue());
  }

  SmallVector<AxisRefAttr> axisRefs;
  mlir::MLIRContext* ctx = mesh.getContext();
  for (auto& [axisName, preSizes] : axisNameToPreSizes) {
    if (preSizes.size() == 2) {
      // Full axis
      axisRefs.push_back(AxisRefAttr::get(ctx, axisName));
      continue;
    }
    llvm::sort(preSizes);
    preSizes.erase(llvm::unique(preSizes), preSizes.end());
    for (int64_t i = 0; i < preSizes.size() - 1; ++i) {
      int64_t preSize = preSizes[i];
      int64_t size = preSizes[i + 1] / preSize;
      axisRefs.push_back(AxisRefAttr::get(
          ctx, axisName, SubAxisInfoAttr::get(ctx, preSize, size)));
    }
  }

  return axisRefs;
}

namespace {

// Check if the func result is meant for Shardy.
bool isFuncResultForShardy(FuncOp func, int64_t resultIndex) {
  if (func.getResultAttr(resultIndex, mlir::sdy::kShardingAttr)) {
    return true;
  }
  Operation* definingOp =
      mlir::sdy::getBodyTerminatorOperand(func, resultIndex).getDefiningOp();
  if (!definingOp) {
    return false;
  }
  auto customCall = mlir::dyn_cast<CustomCallOp>(definingOp);
  if (!customCall) {
    return false;
  }
  return customCall.getCallTargetName() == sdy::kFuncResultShardingTargetName;
}

// Check if the func result shardings are all mhlo shardings for GSPMD.
bool areFuncResultShardingsForGspmd(FuncOp func) {
  for (int64_t resultIndex = 0; resultIndex < func.getNumResults();
       ++resultIndex) {
    if (func.getResultAttr(resultIndex, sdy::kXlaShardingAttr) &&
        !isFuncResultForShardy(func, resultIndex)) {
      return true;
    }
  }
  return false;
}

}  // namespace

bool hasGspmdAttrsOrOps(mlir::ModuleOp module) {
  for (auto func : module.getOps<mlir::func::FuncOp>()) {
    // If Shardy is enabled, we will have added `sdy.sharding`s, on the inputs
    // and outputs of the main function, so no point of checking it. Could
    // even get false positives as we've previously seen where IFRT was once
    // adding replicated `mhlo.sharding`s on all the inputs/outputs.
    if (func.getSymName() != "main") {
      for (int64_t argIndex = 0; argIndex < func.getNumArguments();
           ++argIndex) {
        if (func.getArgAttr(argIndex, sdy::kXlaShardingAttr) &&
            !func.getArgAttr(argIndex, mlir::sdy::kShardingAttr) &&
            !hasKey(sdy::getFuncArgFrontendAttrs(func, argIndex),
                    xla::ToStringRef(HloSharding::kShardingFrontendAttrName))) {
          return true;
        }
      }
    }
    // We check for the module level kOutTupleShardings attribute because there
    // are cases where Shardy shardings are not added to the results of
    // XlaCallModule function. This is likely acceptable as these functions are
    // intended to be inlined. If kOutTupleShardings is set, it indicates that
    // we have added support for Shardy shardings on the wrapper main in tf2xla.
    if (!hasKey(sdy::getFrontendAttrs(module), sdy::kOutTupleShardings) &&
        areFuncResultShardingsForGspmd(func)) {
      return true;
    }
    bool hasGspmd = false;
    // Check the func for a `Sharding` custom call.
    func->walk([&hasGspmd](mlir::stablehlo::CustomCallOp customCall) {
      if (customCall.getCallTargetName() ==
              sdy::kShardingCustomCallTargetName &&
          customCall->hasAttr(sdy::kXlaShardingAttr) &&
          !customCall->hasAttr(mlir::sdy::kShardingAttr) &&
          !hasFrontendAttr(
              customCall,
              xla::ToStringRef(HloSharding::kShardingFrontendAttrName))) {
        hasGspmd = true;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
    if (hasGspmd) {
      return true;
    }
  }
  return false;
}

bool hasShardyMesh(mlir::ModuleOp module) {
  return !module.getOps<mlir::sdy::MeshOp>().empty();
}

namespace {
// Returns the first non-maximal mesh on the result shardings, if there is
// one. Otherwise returns `std::nullopt`.
// TODO(enver): Use a common helper that takes an std::function to get the
// sharding given an index.
std::optional<Attribute> getMeshOrRefOnResults(
    mlir::func::FuncOp funcOp, const mlir::SymbolTable& symbolTable) {
  for (int64_t resultNum = 0; resultNum < funcOp.getNumResults(); ++resultNum) {
    if (mlir::sdy::TensorShardingAttr sdySharding =
            mlir::sdy::getFuncResultSharding(funcOp, resultNum);
        sdySharding && !sdySharding.getMesh(symbolTable).isMaximal()) {
      return std::make_optional(sdySharding.getMeshOrRef());
    }
  }
  return std::nullopt;
}
}  // namespace

mlir::sdy::TensorShardingPerValueAttr getFuncResultShardings(
    mlir::func::CallOp callOp, mlir::func::FuncOp funcOp,
    const mlir::SymbolTable& symbolTable) {
  std::optional<mlir::Attribute> meshOrRef =
      getMeshOrRefOnResults(funcOp, symbolTable);
  if (!meshOrRef) {
    return nullptr;
  }
  SmallVector<mlir::sdy::TensorShardingAttr> resultShardings;
  resultShardings.reserve(funcOp.getNumResults());
  for (int64_t resultNum = 0; resultNum < funcOp.getNumResults(); ++resultNum) {
    mlir::sdy::TensorShardingAttr sdySharding =
        mlir::sdy::getFuncResultSharding(funcOp, resultNum);
    resultShardings.push_back(
        sdySharding ? sdySharding
                    : mlir::sdy::TensorShardingAttr::getFullyOpen(
                          funcOp.getContext(),
                          mlir::sdy::getTensorRank(callOp.getResult(resultNum)),
                          *meshOrRef));
  }
  return mlir::sdy::TensorShardingPerValueAttr::get(funcOp.getContext(),
                                                    resultShardings);
}

}  // namespace sdy
}  // namespace xla
