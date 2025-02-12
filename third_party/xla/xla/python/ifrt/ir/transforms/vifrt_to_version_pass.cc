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

#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
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
    if (version_ref.empty())
      return emitError(op->getLoc())
             << "No target version specified; must be of the form `#.#.#`";
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
  } else if (auto func_type = llvm::dyn_cast<VifrtFunctionV1Type>(type)) {
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
  } else if (auto dense_array_attr =
                 llvm::dyn_cast<mlir::DenseArrayAttr>(attr)) {
    return isLegalType(dense_array_attr.getElementType(), version);
  } else if (auto dict_attr = llvm::dyn_cast<mlir::DictionaryAttr>(attr)) {
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
  } else if (auto array_mapping_attr =
                 llvm::dyn_cast<VifrtArrayMappingV1Attr>(attr)) {
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
    populateVifrtToVersionPatterns(&patterns_, &converter, ctx);
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
      module_op->emitError() << "failed to convert VIFRT version " << version;
      return signalPassFailure();
    }
  }

 private:
  VifrtToVersionConverter converter;
  mlir::FrozenRewritePatternSet patterns;
};

}  // namespace

void populateVifrtToVersionPatterns(mlir::RewritePatternSet* patterns,
                                    mlir::TypeConverter* converter,
                                    mlir::MLIRContext* context) {
  // This is where conversion patterns between op versions will be added when
  // needed.
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateVifrtToVersionPass(
    VifrtToVersionPassOptions options) {
  return std::make_unique<VifrtToVersionPass>(options);
}

}  // namespace ifrt
}  // namespace xla
