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
#include <string>
#include <type_traits>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: export
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/map_ifrt_to_vifrt.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/vifrt_dialect.h"

// Tag to be passed to `-debug-only` argument so that the mlir opt tool prints
// the debug info for this pass.
#define DEBUG_TYPE "ifrt-compat-passes"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTLEGALIZETOVIFRTPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// IFRT to VIFRT attributes
//===----------------------------------------------------------------------===//

// Returns the VIFRT attribute name for the given `NamedAttribute`.
// Cases:
//  1) If the attribute is from the IFRT dialect (e.g., `ifrt.donated`), the
//     name will be converted to the VIFRT dialect (e.g., `vifrt.donated`).
//  2) If the attribute is from the builtin dialect then the name will not be
//     changed.
//  3) Otherwise, a mlir::failure is returned.
mlir::FailureOr<mlir::StringAttr> getAttrNameFromIfrtToVifrt(
    mlir::NamedAttribute attr) {
  auto attr_name = attr.getName();
  if (auto dialect = attr.getNameDialect()) {
    if (dialect->getNamespace() == IfrtDialect::getDialectNamespace()) {
      std::string name_without_dialect = attr_name.str();
      auto dot_pos = name_without_dialect.find('.');
      if (dot_pos != std::string::npos) {
        return mlir::StringAttr::get(
            attr.getValue().getContext(),
            absl::StrCat(VifrtDialect::getDialectNamespace().str(), ".",
                         name_without_dialect.substr(dot_pos + 1)));
      } else {
        return mlir::failure();
      }
    } else if (dialect->getNamespace() !=
               mlir::BuiltinDialect::getDialectNamespace()) {
      return mlir::failure();
    }
  }
  return attr_name;
}

// Returns true if the given `Attribute` is from the IFRT or builtin dialect.
bool isBuiltinOrIfrtAttr(mlir::Attribute attr) {
  auto dialect_namespace = attr.getDialect().getNamespace();
  return dialect_namespace == mlir::BuiltinDialect::getDialectNamespace() ||
         dialect_namespace == IfrtDialect::getDialectNamespace();
}

// Generic conversions; when there's a 1:1 mapping from an IFRT to a VIFRT
// attribute, and passthrough for a subset of the builtin attributes.
mlir::Attribute convertGeneric(mlir::Attribute ifrt_attr,
                               const mlir::TypeConverter* type_converter) {
  LLVM_DEBUG(llvm::dbgs() << "Convert generic attribute: " << ifrt_attr
                          << '\n');
  if (!isBuiltinOrIfrtAttr(ifrt_attr)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << ifrt_attr << '\n');
    return {};
  }
  // Ordered from most constrained to least constrained.
  if (auto attr = llvm::dyn_cast<IfrtDevicesAttr>(ifrt_attr)) {
    return VifrtDevicesV1Attr::get(attr.getContext(), attr.getIds());
  }
  if (auto attr = llvm::dyn_cast<IfrtShardingParamAttr>(ifrt_attr)) {
    return VifrtShardingParamV1Attr::get(attr.getContext(), attr.getSharding());
  }
  if (auto attr = llvm::dyn_cast<IfrtUnspecifiedShardingAttr>(ifrt_attr)) {
    return VifrtUnspecifiedShardingV1Attr::get(attr.getContext());
  }
  if (auto attr = llvm::dyn_cast<IfrtIntervalAttr>(ifrt_attr)) {
    return VifrtIntervalV1Attr::get(attr.getContext(), attr.getStart(),
                                    attr.getEnd(), attr.getStep());
  }
  if (auto attr = llvm::dyn_cast<IfrtMappingAttr>(ifrt_attr)) {
    auto from_shards = llvm::dyn_cast_or_null<VifrtIntervalV1Attr>(
        convertGeneric(attr.getFromShards(), type_converter));
    auto to_shards = llvm::dyn_cast_or_null<VifrtIntervalV1Attr>(
        convertGeneric(attr.getToShards(), type_converter));
    if (!from_shards || !to_shards) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << attr << '\n');
      return {};
    }
    return VifrtMappingV1Attr::get(attr.getContext(), from_shards, to_shards);
  }
  if (auto attr = llvm::dyn_cast<IfrtArrayMappingAttr>(ifrt_attr)) {
    llvm::SmallVector<mlir::Attribute> vifrt_mappings;
    vifrt_mappings.reserve(attr.getMappings().size());
    for (auto mapping : attr.getMappings()) {
      auto vifrt_mapping = llvm::dyn_cast_or_null<VifrtMappingV1Attr>(
          convertGeneric(mapping, type_converter));
      if (!vifrt_mapping) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << attr << '\n');
        return {};
      }
      vifrt_mappings.push_back(vifrt_mapping);
    }
    return VifrtArrayMappingV1Attr::get(
        attr.getContext(), attr.getInArrayIndex(), attr.getOutArrayIndex(),
        mlir::ArrayAttr::get(attr.getContext(), vifrt_mappings));
  }
  if (auto attr = llvm::dyn_cast<mlir::TypeAttr>(ifrt_attr)) {
    auto vifrt_type = type_converter->convertType(attr.getValue());
    if (!vifrt_type) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << attr << '\n');
      return {};
    }
    return VifrtTypeV1Attr::get(attr.getContext(), vifrt_type);
  }
  // A subset of builtin attributes are allowed to not be versioned. They are
  // assumed to be stable, and by not implementing versioning for them, we can
  // avoid implementing versioning for all builtin types.
  if (mlir::isa<mlir::UnitAttr, mlir::BoolAttr, mlir::IntegerAttr,
                mlir::FloatAttr, mlir::StringAttr, mlir::FlatSymbolRefAttr,
                mlir::SymbolRefAttr, mlir::DenseIntOrFPElementsAttr>(
          ifrt_attr)) {
    return ifrt_attr;
  }
  if (auto attr = llvm::dyn_cast<mlir::ArrayAttr>(ifrt_attr)) {
    llvm::SmallVector<mlir::Attribute> vifrt_attrs;
    vifrt_attrs.reserve(attr.getValue().size());
    for (auto ifrt_attr : attr.getValue()) {
      if (auto vifrt_attr = convertGeneric(ifrt_attr, type_converter)) {
        vifrt_attrs.push_back(vifrt_attr);
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << attr << '\n');
        return {};
      }
    }
    return mlir::ArrayAttr::get(attr.getContext(), vifrt_attrs);
  }
  if (auto attr = llvm::dyn_cast<mlir::DenseArrayAttr>(ifrt_attr)) {
    // Only dense array attributes with the following element types are allowed.
    // Other element types are not allowed because we would have to convert the
    // raw data. One should use `ArrayAttr` instead for such arrays.
    if (mlir::isa<mlir::IntegerType, mlir::FloatType>(attr.getElementType())) {
      return ifrt_attr;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << attr << '\n');
      return {};
    }
  }
  if (auto attr = llvm::dyn_cast<mlir::DictionaryAttr>(ifrt_attr)) {
    llvm::SmallVector<mlir::NamedAttribute> vifrt_attrs;
    vifrt_attrs.reserve(attr.getValue().size());
    for (auto named_attr : attr.getValue()) {
      auto attr_name = getAttrNameFromIfrtToVifrt(named_attr);
      if (mlir::failed(attr_name)) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << ifrt_attr << '\n');
        return {};
      }
      if (auto vifrt_attr =
              convertGeneric(named_attr.getValue(), type_converter)) {
        vifrt_attrs.push_back({attr_name.value(), vifrt_attr});
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << ifrt_attr << '\n');
        return {};
      }
    }
    return mlir::DictionaryAttr::get(attr.getContext(), vifrt_attrs);
  }

  LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << ifrt_attr << '\n');
  return {};
}

//===----------------------------------------------------------------------===//
// IFRT to VIFRT types
//===----------------------------------------------------------------------===//

class IfrtToVifrtTypeConverter : public VifrtTypeConverterBuiltin {
 public:
  IfrtToVifrtTypeConverter() : VifrtTypeConverterBuiltin() {
    addConversion([](mlir::Type type) -> mlir::Type {
      // We currently rely on the builtin types being stable, and thus we do not
      // convert builtin types to VIFRT types.
      if (type.getDialect().getNamespace() ==
              VifrtDialect::getDialectNamespace() ||
          type.getDialect().getNamespace() ==
              mlir::BuiltinDialect::getDialectNamespace()) {
        return type;
      }
      LLVM_DEBUG(llvm::dbgs() << "Invalid type: " << type << '\n');
      return {};
    });
    addConversion([&](IfrtArrayType array) -> mlir::Type {
      mlir::StringAttr memory_kind_attr = array.getMemoryKindAttr();
      if (!memory_kind_attr) {
        // Use a default string value to indicate that memory kind was not set.
        memory_kind_attr =
            mlir::StringAttr::get(array.getContext(), kVifrtDefaultString);
      };
      mlir::StringAttr layout_attr = array.getLayoutAttr();
      if (!layout_attr) {
        // Use a default string value to indicate that layout was not set.
        layout_attr =
            mlir::StringAttr::get(array.getContext(), kVifrtDefaultString);
      }
      auto sharding_attr = convertGeneric(array.getShardingAttr(), this);
      if (!sharding_attr) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert sharding: "
                                << array.getShardingAttr() << '\n');
        return {};
      }
      auto devices_attr = llvm::dyn_cast_or_null<VifrtDevicesV1Attr>(
          convertGeneric(array.getDevicesAttr(), this));
      if (!devices_attr) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert devices: "
                                << array.getDevicesAttr() << '\n');
        return {};
      }
      return VifrtArrayV1Type::get(array.getContext(), array.getShape(),
                                   sharding_attr, devices_attr,
                                   memory_kind_attr, layout_attr);
    });
    addConversion([](IfrtControlType type) -> mlir::Type {
      return VifrtControlV1Type::get(type.getContext());
    });
    addBuiltinToVifrtConversions();
  }
};

//===----------------------------------------------------------------------===//
// IFRT to VIFRT operations
//===----------------------------------------------------------------------===//

// Unlike IFRT, VIFRT does not have default attributes.
// This function adds the default attributes to VIFRT ops.
template <typename IfrtOpTy>
mlir::LogicalResult addDefaultAttrs(
    const mlir::OpConversionPattern<IfrtOpTy>& pattern, IfrtOpTy ifrt_op,
    llvm::SmallVector<mlir::NamedAttribute>& vifrt_attrs) {
  mlir::Builder builder(pattern.getContext());
  auto add_default_attr = [&](mlir::StringRef vifrt_name,
                              mlir::Attribute ifrt_attr) {
    vifrt_attrs.emplace_back(
        mlir::StringAttr::get(pattern.getContext(), vifrt_name),
        convertGeneric(ifrt_attr, pattern.getTypeConverter()));
  };

  if constexpr (std::is_same<IfrtOpTy, ReshardOp>::value ||
                std::is_same<IfrtOpTy, CopyArraysOp>::value ||
                std::is_same<IfrtOpTy, RemapArraysOp>::value) {
    if (!ifrt_op.getDonatedAttr()) {
      add_default_attr("donated", builder.getBoolAttr(false));
    }
  } else if constexpr (std::is_same<IfrtOpTy, CallOp>::value ||
                       std::is_same<IfrtOpTy, CallLoadedExecutableOp>::value) {
    if (!ifrt_op.getIoAliasesAttr()) {
      add_default_attr("io_aliases", builder.getArrayAttr({}));
    }
    if (!ifrt_op.getDonatedInputIndicesAttr()) {
      add_default_attr("donated_input_indices",
                       builder.getDenseI32ArrayAttr({}));
    }
  } else if constexpr (std::is_same<IfrtOpTy, mlir::func::FuncOp>::value) {
    if (!ifrt_op.getSymVisibilityAttr())
      add_default_attr(
          "sym_visibility",
          mlir::StringAttr::get(pattern.getContext(), kVifrtDefaultString));
    if (!ifrt_op.getArgAttrsAttr())
      add_default_attr("arg_attrs",
                       mlir::ArrayAttr::get(pattern.getContext(), {}));
    if (!ifrt_op.getResAttrsAttr())
      add_default_attr("res_attrs",
                       mlir::ArrayAttr::get(pattern.getContext(), {}));
  }
  return mlir::success();
}

// Generic op conversions. All ops present in IFRT IR (including
// mlir::func::FuncOp, mlir::func::CallOp, mlir::func::ReturnOp) have a 1:1
// mapping to VIFRT ops.
template <typename IfrtOpTy>
class IfrtToVifrtOpConverter : public mlir::OpConversionPattern<IfrtOpTy> {
 public:
  using mlir::OpConversionPattern<IfrtOpTy>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      IfrtOpTy ifrt_op, typename IfrtOpTy::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const final {
    // Convert the IFRT result types to VIFRT types.
    llvm::SmallVector<mlir::Type> vifrt_types;
    if (mlir::failed(this->getTypeConverter()->convertTypes(
            ifrt_op->getResultTypes(), vifrt_types))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed IFRT to VIFRT type conversion\n");
      return mlir::failure();
    }

    // Convert the IFRT attributes to VIFRT attributes.
    llvm::SmallVector<mlir::NamedAttribute> vifrt_attrs;
    if (mlir::failed(addDefaultAttrs(*this, ifrt_op, vifrt_attrs))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to convert default IFRT attributes to VIFRT\n");
      return mlir::failure();
    }

    // Special case ops.
    llvm::DenseSet<mlir::StringAttr> already_converted_attrs;
    if constexpr (std::is_same<IfrtOpTy, CallOp>::value) {
      auto call_op = static_cast<CallOp>(ifrt_op);
      // Convert the callee from SymbolRefAttr to StringAttr so that DCE
      // can remove the atom programs, which have independently legalized to
      // VHLO. Manually to the conversion by merging RootReference and
      // NestedReferences to avoid string escaping.
      std::string symbol_ref_str = absl::StrCat(
          "@", call_op.getCalleeAttr().getRootReference().getValue().str(),
          absl::StrJoin(
              call_op.getCalleeAttr().getNestedReferences(), "",
              [](std::string* out, const mlir::FlatSymbolRefAttr& symbol_ref) {
                absl::StrAppend(out, "::@", symbol_ref.getValue().str());
              }));
      vifrt_attrs.push_back(
          {call_op.getCalleeAttrName(),
           mlir::StringAttr::get(call_op.getContext(), symbol_ref_str)});
      already_converted_attrs.insert(call_op.getCalleeAttrName());
    }

    for (mlir::NamedAttribute ifrt_attr : ifrt_op->getAttrs()) {
      // Skip special case attributes, which have already been converted.
      if (already_converted_attrs.contains(ifrt_attr.getName())) {
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "Converting " << ifrt_attr.getName() << ", "
                              << ifrt_attr.getValue() << '\n');
      auto attr_name = getAttrNameFromIfrtToVifrt(ifrt_attr);
      if (mlir::failed(attr_name)) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert attribute name: "
                                << ifrt_attr.getName() << ", "
                                << ifrt_attr.getValue() << '\n');
        return mlir::failure();
      }
      if (auto vifrt_attr =
              convertGeneric(ifrt_attr.getValue(), this->getTypeConverter())) {
        vifrt_attrs.push_back({attr_name.value(), vifrt_attr});
      } else {
        return mlir::failure();
      }
    }

    // The operands have already been converted to VIFRT by the dialect
    // conversion infrastructure.
    mlir::ValueRange vifrt_operands = adaptor.getOperands();

    // Convert the IFRT op to a VIFRT equivalent op.
    IfrtToVifrtOp<IfrtOpTy> vifrt_op = rewriter.create<IfrtToVifrtOp<IfrtOpTy>>(
        ifrt_op.getLoc(), vifrt_types, vifrt_operands, vifrt_attrs);

    // Convert the IFRT region types to VIFRT region types.
    for (auto [ifrt_region, vifrt_region] :
         llvm::zip(ifrt_op->getRegions(), vifrt_op->getRegions())) {
      rewriter.inlineRegionBefore(ifrt_region, vifrt_region,
                                  vifrt_region.end());
      if (mlir::failed(rewriter.convertRegionTypes(
              &vifrt_region, *this->getTypeConverter(),
              /*entryConversion=*/nullptr)))
        return mlir::failure();
    }
    rewriter.replaceOp(ifrt_op, vifrt_op);
    return mlir::success();
  }
};

template <typename... IfrtOpTypes>
void populateIfrtToVifrtPatterns(mlir::RewritePatternSet* patterns,
                                 mlir::TypeConverter* converter,
                                 mlir::MLIRContext* context) {
  patterns->add<IfrtToVifrtOpConverter<IfrtOpTypes>...>(*converter, context);
}

}  // namespace

struct IfrtLegalizeToVifrtPass
    : public impl::IfrtLegalizeToVifrtPassBase<IfrtLegalizeToVifrtPass> {
  mlir::LogicalResult initialize(mlir::MLIRContext* context) override {
    target = std::make_shared<mlir::ConversionTarget>(*context);
    target->addIllegalDialect<IfrtDialect>();
    target->addLegalDialect<VifrtDialect>();
    target->addDynamicallyLegalOp<mlir::func::FuncOp>(
        [](mlir::func::FuncOp func_op) {
          // FuncOps that are not IFRT functions are either VIFRT functions or
          // legal because they will be removed by DCE.
          if (func_op->hasAttr(kIfrtFunctionAttrName)) {
            return false;
          } else {
            return true;
          }
        });
    target->addDynamicallyLegalOp<mlir::func::CallOp>(
        [](mlir::func::CallOp call_op) {
          // CallOps of non IFRT functions are legal because the funcs are
          // either VIFRT functions or they will be removed by DCE.
          auto func_op =
              mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
                  call_op, call_op.getCalleeAttr());
          if (func_op->hasAttr(kIfrtFunctionAttrName)) {
            return false;
          } else {
            return true;
          }
        });
    target->addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [](mlir::func::ReturnOp return_op) {
          if (return_op->getParentOp()->hasAttr(kIfrtFunctionAttrName) ||
              return_op->getParentOp()->hasAttr(kVifrtFunctionAttrName)) {
            return false;
          }
          // ReturnOps of non-IFRT functions are legal because they will be
          // removed by DCE.
          return true;
        });

    mlir::RewritePatternSet patterns_(context);
    // Populate the patterns for the generic IFRT op to VIFRT op conversions.
    populateIfrtToVifrtPatterns(&patterns_, &converter, context);
    patterns = std::move(patterns_);

    return mlir::success();
  }

  void runOnOperation() override {
    if (mlir::failed(
            applyPartialConversion(getOperation(), *target, patterns))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed partial conversion\n");
      return signalPassFailure();
    }
  }

 private:
  IfrtToVifrtTypeConverter converter;
  mlir::FrozenRewritePatternSet patterns;
  std::shared_ptr<mlir::ConversionTarget> target;
};

void populateIfrtToVifrtPatterns(mlir::RewritePatternSet* patterns,
                                 mlir::TypeConverter* converter,
                                 mlir::MLIRContext* context) {
  populateIfrtToVifrtPatterns<
#define GET_OP_LIST
#include "xla/python/ifrt/ir/ifrt_ops.cc.inc"
      , mlir::func::CallOp, mlir::func::FuncOp, mlir::func::ReturnOp>(
      patterns, converter, context);
}

}  // namespace ifrt
}  // namespace xla
