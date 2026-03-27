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

#include <cstddef>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: export
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/version.h"
#include "xla/python/ifrt/ir/vifrt_dialect.h"

// Tag to be passed to `-debug-only` argument so that the mlir opt tool prints
// the debug info for this pass.
#define DEBUG_TYPE "ifrt-compat-passes"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_VIFRTTOVERSIONPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

namespace {

template <typename TypeOrAttr>
bool isFromBuiltinDialect(TypeOrAttr type_or_attr) {
  return type_or_attr.getDialect().getNamespace() ==
         mlir::BuiltinDialect::getDialectNamespace();
}

class VifrtToVersionConverter : public mlir::TypeConverter {
 public:
  VifrtToVersionConverter() : mlir::TypeConverter() {
    // Currently, there are no conversions to be done for the VIFRT types.
    // The converter just validates that the types are from the VIFRT or
    // Builtin dialects.
    addConversion([](mlir::Type type) -> mlir::Type {
      if (type.getDialect().getNamespace() ==
              VifrtDialect::getDialectNamespace() ||
          isFromBuiltinDialect(type)) {
        return type;
      }
      LLVM_DEBUG(llvm::dbgs() << "Invalid type: " << type << '\n');
      return {};
    });
  }
};

// Validate the requested target version. Emit errors if: 1) the version is not
// in a valid format, 2) is less than the minimum supported version, or 3) is
// greater than the current version.
mlir::FailureOr<Version> validateTargetVersion(llvm::StringRef version_ref,
                                               mlir::Operation* op) {
  auto version_or = Version::fromString(version_ref);
  if (mlir::failed(version_or)) {
    if (version_ref.empty()) {
      return emitError(op->getLoc())
             << "No target version specified; must be of the form `#.#.#`";
    }
    return emitError(op->getLoc())
           << "Invalid target version argument '" << version_ref
           << "'; version must be of the form `#.#.#`.";
  }
  Version version = *version_or;
  if (version < Version::getMinimumVersion()) {
    return emitError(op->getLoc())
           << "target version " << version << " is less than minimum version "
           << Version::getMinimumVersion();
  }
  if (Version::getCurrentVersion() < version) {
    return emitError(op->getLoc()) << "target version " << version
                                   << " is greater than current version "
                                   << Version::getCurrentVersion();
  }
  return version;
}

// Can be used for attrs, types, and ops as they all define `getMinVersion` and
// `getMaxVersion`.
template <typename VersionedInterface>
bool isLegalVersion(VersionedInterface& interface, const Version& version) {
  return interface.getMinVersion() <= version &&
         version <= interface.getMaxVersion();
}

// Forward declaration because `isLegalType` and `isLegalAttribute` are
// mutually recursive.
bool isLegalAttribute(mlir::Attribute attr, const Version& version);

bool isLegalType(mlir::Type type, const Version& version) {
  if (isFromBuiltinDialect(type)) {
    return true;
  }
  auto ver_type_interface = llvm::dyn_cast<VifrtVersionedTypeInterface>(type);
  if (!ver_type_interface || !isLegalVersion(ver_type_interface, version)) {
    LLVM_DEBUG(llvm::dbgs() << "failed to convert type " << type
                            << " to version " << version << '\n');
    return false;
  }
  if (auto array_type = llvm::dyn_cast<VifrtArrayV1Type>(type)) {
    return isLegalAttribute(array_type.getShardingAttr(), version);
  }
  if (auto func_type = llvm::dyn_cast<VifrtFunctionV1Type>(type)) {
    auto is_legal_type_fn = [&](mlir::Type type) {
      return isLegalType(type, version);
    };
    return llvm::all_of(func_type.getInputs(), is_legal_type_fn) &&
           llvm::all_of(func_type.getOutputs(), is_legal_type_fn);
  }
  return true;
}

bool isLegalAttribute(mlir::Attribute attr, const Version& version) {
  // Recursively check all the elements of container attributes.
  if (auto array_attr = llvm::dyn_cast<mlir::ArrayAttr>(attr)) {
    return llvm::all_of(array_attr.getValue(), [&](mlir::Attribute entry) {
      return isLegalAttribute(entry, version);
    });
  }
  if (auto dense_array_attr = llvm::dyn_cast<mlir::DenseArrayAttr>(attr)) {
    return isLegalType(dense_array_attr.getElementType(), version);
  }
  if (auto dict_attr = llvm::dyn_cast<mlir::DictionaryAttr>(attr)) {
    return llvm::all_of(dict_attr.getValue(), [&](const auto& entry) {
      return isLegalAttribute(entry.getValue(), version);
    });
  }
  // Check if it is an allowed attribute from the builtin dialect.
  if (mlir::isa<mlir::UnitAttr, mlir::BoolAttr, mlir::IntegerAttr,
                mlir::FloatAttr, mlir::StringAttr, mlir::FlatSymbolRefAttr,
                mlir::SymbolRefAttr, mlir::DenseIntOrFPElementsAttr>(attr)) {
    return true;
  }
  // Check if it is an allowed VIFRT attribute.
  auto ver_attr_interface = llvm::dyn_cast<VifrtVersionedAttrInterface>(attr);
  if (!ver_attr_interface || !isLegalVersion(ver_attr_interface, version)) {
    LLVM_DEBUG(llvm::dbgs() << "failed to convert attribute " << attr
                            << " to version " << version << '\n');
    return false;
  }
  if (auto type_attr = llvm::dyn_cast<VifrtTypeV1Attr>(attr)) {
    return isLegalType(type_attr.getValue(), version);
  }
  if (auto array_mapping_attr = llvm::dyn_cast<VifrtArrayMappingV1Attr>(attr)) {
    return llvm::all_of(array_mapping_attr.getMappings().getValue(),
                        [&](const auto& mapping) {
                          return isLegalAttribute(mapping, version);
                        });
  }
  return true;
}

bool isLegalOperation(mlir::Operation* op, const Version& version) {
  auto ver_op_interface = llvm::dyn_cast<VifrtVersionedOpInterface>(op);
  if (!ver_op_interface || !isLegalVersion(ver_op_interface, version)) {
    LLVM_DEBUG(llvm::dbgs() << "failed to convert operation " << op
                            << " to version " << version << '\n');
    return false;
  }
  // Verify if all the attributes are legal.
  if (!llvm::all_of(op->getAttrs(), [&](const mlir::NamedAttribute& attr) {
        return isLegalAttribute(attr.getValue(), version);
      })) {
    return false;
  }
  // Verify if all the types are legal.
  auto is_legal_type_fn = [&](mlir::Type type) {
    return isLegalType(type, version);
  };
  if (!llvm::all_of(op->getOperandTypes(), is_legal_type_fn) ||
      !llvm::all_of(op->getResultTypes(), is_legal_type_fn)) {
    return false;
  }

  return true;
}

struct VifrtToVersionPass
    : public impl::VifrtToVersionPassBase<VifrtToVersionPass> {
 public:
  using impl::VifrtToVersionPassBase<
      VifrtToVersionPass>::VifrtToVersionPassBase;

  mlir::LogicalResult initialize(mlir::MLIRContext* ctx) override {
    mlir::RewritePatternSet patterns_(ctx);
    mlir::FailureOr<Version> version_or = Version::fromString(target_version);
    if (mlir::failed(version_or)) {
      return mlir::failure();
    }
    populateVifrtToVersionPatterns(&patterns_, &converter, *version_or, ctx);
    patterns = std::move(patterns_);
    return mlir::success();
  }

  void runOnOperation() override {
    auto module_op = getOperation();
    auto version_or = validateTargetVersion(target_version, module_op);
    if (mlir::failed(version_or)) {
      return signalPassFailure();
    }
    Version version = *version_or;
    mlir::ConversionTarget conversion_target(getContext());
    conversion_target.addDynamicallyLegalDialect<VifrtDialect>(
        [&version](mlir::Operation* op) {
          return isLegalOperation(op, version);
        });

    // Conversions within VHLO may fail if new features or ops are used.
    if (mlir::failed(applyPartialConversion(getOperation(), conversion_target,
                                            patterns))) {
      module_op->emitError()
          << "failed to convert to VIFRT version " << version;
      return signalPassFailure();
    }
  }

 private:
  VifrtToVersionConverter converter;
  mlir::FrozenRewritePatternSet patterns;
};

mlir::FailureOr<mlir::Attribute> convertShardingAttr(
    mlir::Attribute sharding_attr, const Version& version) {
  // Upgrade VifrtShardingParamV1Attr to VifrtShardingParamV2Attr.
  if (auto sp_attr = llvm::dyn_cast<VifrtShardingParamV1Attr>(sharding_attr);
      sp_attr != nullptr) {
    if (sp_attr.getMaxVersion() < version) {
      return VifrtShardingParamV2Attr::get(sp_attr.getContext(),
                                           sp_attr.getSharding());
    }
  }

  // Downgrade VifrtShardingParamV2Attr to VifrtShardingParamV1Attr.
  if (auto sp_attr = llvm::dyn_cast<VifrtShardingParamV2Attr>(sharding_attr);
      sp_attr != nullptr) {
    if (version < sp_attr.getMinVersion()) {
      ShardingParam sharding_param = sp_attr.getSharding();
      if (!sharding_param.unreduced_axes().empty()) {
        // Cannot convert to VifrtShardingParamV1Attr because of the unreduced
        // axes.
        return mlir::failure();
      }
      return VifrtShardingParamV1Attr::get(sp_attr.getContext(),
                                           sp_attr.getSharding());
    }
  }

  return mlir::failure();
}

// Conversion pattern for VIFRT types. Applies to all VIFRT ops, including
// FuncOpV*, CallOpV*, and ReturnOpV*.
struct VifrtTypeConversionPattern : public mlir::ConversionPattern {
  VifrtTypeConversionPattern(mlir::TypeConverter& converter,
                             mlir::MLIRContext* context, Version version)
      : mlir::ConversionPattern(converter, MatchAnyOpTypeTag(),
                                /*benefit=*/1, context),
        version(version) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::Operation* op, llvm::ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter& rewriter) const override {
    // Only convert ops from the VIFRT dialect.
    if (op->getDialect()->getNamespace() !=
        VifrtDialect::getDialectNamespace()) {
      return mlir::failure();
    }

    // Convert the attributes.
    llvm::SmallVector<mlir::NamedAttribute> new_attrs;
    for (mlir::NamedAttribute named_attr : op->getAttrs()) {
      if (auto type_attr =
              llvm::dyn_cast<VifrtTypeV1Attr>(named_attr.getValue())) {
        new_attrs.push_back(
            {named_attr.getName(),
             VifrtTypeV1Attr::get(
                 type_attr.getContext(),
                 this->getTypeConverter()->convertType(type_attr.getValue()))});
        continue;
      }
      if (auto new_attr = convertShardingAttr(named_attr.getValue(), *version);
          mlir::succeeded(new_attr)) {
        new_attrs.push_back({named_attr.getName(), *new_attr});
        continue;
      }
      new_attrs.push_back(named_attr);
    }

    // Convert the result types.
    llvm::SmallVector<mlir::Type> new_res_types;
    if (mlir::failed(this->getTypeConverter()->convertTypes(
            op->getResultTypes(), new_res_types))) {
      return rewriter.notifyMatchFailure(op, "Failed to convert result types");
    }

    mlir::OperationState state(op->getLoc(), op->getName().getStringRef(),
                               operands, new_res_types, new_attrs,
                               op->getSuccessors());

    // Allocate space for regions if the operation has them
    for (size_t i = 0; i < op->getNumRegions(); ++i) {
      state.addRegion();
    }

    mlir::Operation* new_op = rewriter.create(state);

    for (auto [old_region, new_region] :
         llvm::zip(op->getRegions(), new_op->getRegions())) {
      rewriter.inlineRegionBefore(old_region, new_region, new_region.end());
      if (mlir::failed(rewriter.convertRegionTypes(
              &new_region, *this->getTypeConverter(),
              /*entryConversion=*/nullptr))) {
        return mlir::failure();
      }
    }

    rewriter.replaceOp(op, new_op);

    return mlir::success();
  }

  // Wrapped in an optional because Version is not default constructible.
  std::optional<Version> version;
};

}  // namespace

void populateVifrtToVersionPatterns(mlir::RewritePatternSet* patterns,
                                    mlir::TypeConverter* converter,
                                    Version version,
                                    mlir::MLIRContext* context) {
  // Upgrade/Downgrade between VifrtShardingParamV1Attr and
  // VifrtShardingParamV2Attr. ShardingParam can appear as an attr in:
  // 1) VifrtArrayV*Type, 2) VifrtFunctionV*Type, 3) typed attributes,
  // 4) operations.

  // 1) Convert the types in the VifrtArrayV1Type.
  converter->addConversion([version](VifrtArrayV1Type type) -> mlir::Type {
    mlir::FailureOr<mlir::Attribute> sharding_attr_or =
        convertShardingAttr(type.getShardingAttr(), version);
    if (mlir::failed(sharding_attr_or)) {
      return type;
    }
    return VifrtArrayV1Type::get(
        type.getContext(), type.getShape(), sharding_attr_or.value(),
        type.getDevicesAttr(), type.getMemoryKindAttr(), type.getLayoutAttr());
  });

  // 2) Convert the types in the VifrtFunctionV1Type.
  converter->addConversion([version](VifrtFunctionV1Type type) -> mlir::Type {
    auto convert_types_fn =
        [&version](
            llvm::ArrayRef<mlir::Type> types) -> llvm::SmallVector<mlir::Type> {
      llvm::SmallVector<mlir::Type> new_types;
      new_types.reserve(types.size());
      for (mlir::Type type : types) {
        if (auto array_type = llvm::dyn_cast<VifrtArrayV1Type>(type)) {
          mlir::FailureOr<mlir::Attribute> sharding_attr_or =
              convertShardingAttr(array_type.getShardingAttr(), version);
          if (mlir::failed(sharding_attr_or)) {
            new_types.push_back(type);
          } else {
            new_types.push_back(VifrtArrayV1Type::get(
                array_type.getContext(), array_type.getShape(),
                sharding_attr_or.value(), array_type.getDevicesAttr(),
                array_type.getMemoryKindAttr(), array_type.getLayoutAttr()));
          }
        } else {
          new_types.push_back(type);
        }
      }
      return new_types;
    };

    llvm::SmallVector<mlir::Type> new_inputs =
        convert_types_fn(type.getInputs());
    llvm::SmallVector<mlir::Type> new_outputs =
        convert_types_fn(type.getOutputs());
    return VifrtFunctionV1Type::get(type.getContext(), new_inputs, new_outputs);
  });

  // 3) and 4) Convert the typed attributes and attributes of operations.
  patterns->add<VifrtTypeConversionPattern>(*converter, context, version);
}

}  // namespace ifrt
}  // namespace xla
