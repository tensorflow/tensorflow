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
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: export
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/transforms/map_ifrt_to_vifrt.h"
#include "xla/python/ifrt/ir/transforms/passes.h"
#include "xla/python/ifrt/ir/vifrt_dialect.h"

// Tag to be passed to `-debug-only` argument so that the mlir opt tool prints
// the debug info for this pass.
#define DEBUG_TYPE "ifrt-compat-passes"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_VIFRTLEGALIZETOIFRTPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// VIFRT to IFRT attributes
//===----------------------------------------------------------------------===//

// Returns the IFRT attribute name for the given `NamedAttribute`.
// Cases:
//  1) If the attribute is from the VIFRT dialect (e.g., `vifrt.donated`), the
//     name will be converted to the IFRT dialect (e.g., `ifrt.donated`).
//  2) If the attribute is from the builtin dialect then the name will not be
//     changed.
//  3) Otherwise, a mlir::failure is returned.
mlir::FailureOr<mlir::StringAttr> getAttrNameFromVifrtToIfrt(
    mlir::NamedAttribute attr) {
  auto attr_name = attr.getName();
  if (auto dialect = attr.getNameDialect()) {
    if (dialect->getNamespace() == VifrtDialect::getDialectNamespace()) {
      std::string name_without_dialect = attr_name.str();
      auto dot_pos = name_without_dialect.find('.');
      if (dot_pos != std::string::npos) {
        return mlir::StringAttr::get(
            attr.getValue().getContext(),
            absl::StrCat(IfrtDialect::getDialectNamespace().str(), ".",
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

// Returns true if the given `Attribute` is from the VIFRT or builtin dialect.
bool isBuiltinOrVifrtAttr(mlir::Attribute attr) {
  auto dialect_namespace = attr.getDialect().getNamespace();
  return dialect_namespace == mlir::BuiltinDialect::getDialectNamespace() ||
         dialect_namespace == VifrtDialect::getDialectNamespace();
}

// Generic conversions; when there's a 1:1 mapping from a VIFRT to an IFRT
// attribute, and passthrough for a subset of the builtin attributes.
mlir::Attribute convertGeneric(mlir::Attribute vifrt_attr,
                               const mlir::TypeConverter* type_converter) {
  LLVM_DEBUG(llvm::dbgs() << "Convert generic attribute: " << vifrt_attr
                          << '\n');
  if (!isBuiltinOrVifrtAttr(vifrt_attr)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << vifrt_attr << '\n');
    return {};
  }
  // Ordered from most constrained to least constrained.
  if (auto attr = llvm::dyn_cast<VifrtDevicesV1Attr>(vifrt_attr)) {
    return IfrtDevicesAttr::get(attr.getContext(), attr.getIds());
  }
  if (auto attr = llvm::dyn_cast<VifrtShardingParamV1Attr>(vifrt_attr)) {
    return IfrtShardingParamAttr::get(attr.getContext(), attr.getSharding());
  }
  if (auto attr = llvm::dyn_cast<VifrtUnspecifiedShardingV1Attr>(vifrt_attr)) {
    return IfrtUnspecifiedShardingAttr::get(attr.getContext());
  }
  if (auto attr = llvm::dyn_cast<VifrtIntervalV1Attr>(vifrt_attr)) {
    return IfrtIntervalAttr::get(attr.getContext(), attr.getStart(),
                                 attr.getEnd(), attr.getStep());
  }
  if (auto attr = llvm::dyn_cast<VifrtMappingV1Attr>(vifrt_attr)) {
    auto from_shards = llvm::dyn_cast_or_null<IfrtIntervalAttr>(
        convertGeneric(attr.getFromShards(), type_converter));
    auto to_shards = llvm::dyn_cast_or_null<IfrtIntervalAttr>(
        convertGeneric(attr.getToShards(), type_converter));
    if (!from_shards || !to_shards) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << attr << '\n');
      return {};
    }
    return IfrtMappingAttr::get(attr.getContext(), from_shards, to_shards);
  }
  if (auto attr = llvm::dyn_cast<VifrtArrayMappingV1Attr>(vifrt_attr)) {
    llvm::SmallVector<mlir::Attribute> ifrt_mappings;
    ifrt_mappings.reserve(attr.getMappings().size());
    for (auto mapping : attr.getMappings()) {
      auto ifrt_mapping = llvm::dyn_cast_or_null<IfrtMappingAttr>(
          convertGeneric(mapping, type_converter));
      if (!ifrt_mapping) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << attr << '\n');
        return {};
      }
      ifrt_mappings.push_back(ifrt_mapping);
    }
    return IfrtArrayMappingAttr::get(
        attr.getContext(), attr.getInArrayIndex(), attr.getOutArrayIndex(),
        mlir::ArrayAttr::get(attr.getContext(), ifrt_mappings));
  }
  if (auto attr = llvm::dyn_cast<VifrtTypeV1Attr>(vifrt_attr)) {
    auto ifrt_type = type_converter->convertType(attr.getValue());
    if (!ifrt_type) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << vifrt_attr << '\n');
      return {};
    }
    return mlir::TypeAttr::get(ifrt_type);
  }

  // A subset of builtin attributes are allowed to not be versioned. They are
  // assumed to be stable, and by not implementing versioning for them, we can
  // avoid implementing versioning for all builtin types.
  if (mlir::isa<mlir::UnitAttr, mlir::BoolAttr, mlir::IntegerAttr,
                mlir::FloatAttr, mlir::StringAttr, mlir::FlatSymbolRefAttr,
                mlir::SymbolRefAttr, mlir::DenseIntOrFPElementsAttr>(
          vifrt_attr)) {
    return vifrt_attr;
  }
  if (auto attr = llvm::dyn_cast<mlir::ArrayAttr>(vifrt_attr)) {
    llvm::SmallVector<mlir::Attribute> ifrt_attrs;
    ifrt_attrs.reserve(attr.getValue().size());
    for (auto vifrt_attr : attr.getValue()) {
      if (auto ifrt_attr = convertGeneric(vifrt_attr, type_converter)) {
        ifrt_attrs.push_back(ifrt_attr);
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << vifrt_attr << '\n');
        return {};
      }
    }
    return mlir::ArrayAttr::get(attr.getContext(), ifrt_attrs);
  }
  if (auto attr = llvm::dyn_cast<mlir::DenseArrayAttr>(vifrt_attr)) {
    // Only dense array attributes with the following element types are allowed.
    // Other element types are not allowed because we would have to convert the
    // raw data. One should use `ArrayAttr` instead for such arrays.
    if (mlir::isa<mlir::IntegerType, mlir::FloatType>(attr.getElementType())) {
      return attr;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << vifrt_attr << '\n');
      return {};
    }
  }
  if (auto attr = llvm::dyn_cast<mlir::DictionaryAttr>(vifrt_attr)) {
    llvm::SmallVector<mlir::NamedAttribute> ifrt_attrs;
    ifrt_attrs.reserve(attr.getValue().size());
    for (auto named_attr : attr.getValue()) {
      auto attr_name = getAttrNameFromVifrtToIfrt(named_attr);
      if (mlir::failed(attr_name)) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << vifrt_attr << '\n');
        return {};
      }
      if (auto ifrt_attr =
              convertGeneric(named_attr.getValue(), type_converter)) {
        ifrt_attrs.push_back({attr_name.value(), ifrt_attr});
      } else {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << vifrt_attr << '\n');
        return {};
      }
    }
    return mlir::DictionaryAttr::get(attr.getContext(), ifrt_attrs);
  }

  LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << vifrt_attr << '\n');
  return {};
}

//===----------------------------------------------------------------------===//
// VIFRT to IFRT types
//===----------------------------------------------------------------------===//

class VifrtToIfrtTypeConverter : public VifrtTypeConverterBuiltin {
 public:
  VifrtToIfrtTypeConverter() : VifrtTypeConverterBuiltin() {
    addConversion([](mlir::Type type) -> mlir::Type {
      // We currently rely on the builtin types being stable, and thus we do not
      // convert builtin types to VIFRT types. Therefore, we need to ignore them
      // on the conversion back to IFRT.
      if (type.getDialect().getNamespace() ==
              IfrtDialect::getDialectNamespace() ||
          type.getDialect().getNamespace() ==
              mlir::BuiltinDialect::getDialectNamespace()) {
        return type;
      }
      LLVM_DEBUG(llvm::dbgs() << "Invalid type: " << type << '\n');
      return {};
    });
    addConversion([&](VifrtArrayV1Type array) -> mlir::Type {
      auto sharding_attr = llvm::dyn_cast_or_null<IfrtShardingAttrInterface>(
          convertGeneric(array.getShardingAttr(), this));
      if (!sharding_attr) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert sharding: "
                                << array.getShardingAttr() << '\n');
        return {};
      }
      auto devices_attr = llvm::dyn_cast_or_null<IfrtDevicesAttr>(
          convertGeneric(array.getDevicesAttr(), this));
      if (!devices_attr) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert devices: "
                                << array.getDevicesAttr() << '\n');
        return {};
      }
      auto memory_kind_attr =
          llvm::dyn_cast<mlir::StringAttr>(array.getMemoryKindAttr());
      if (memory_kind_attr && memory_kind_attr.str() == kVifrtDefaultString) {
        // No memory kind was specified.
        memory_kind_attr = nullptr;
      }
      auto layout_attr =
          llvm::dyn_cast<mlir::StringAttr>(array.getLayoutAttr());
      if (layout_attr && layout_attr.str() == kVifrtDefaultString) {
        // No layout was specified.
        layout_attr = nullptr;
      }
      return IfrtArrayType::get(array.getContext(), array.getShape(),
                                sharding_attr, devices_attr, memory_kind_attr,
                                layout_attr);
    });
    addConversion([](VifrtControlV1Type type) -> mlir::Type {
      return IfrtControlType::get(type.getContext());
    });
    addVifrtToBuiltinConversions();
  }
};

//===----------------------------------------------------------------------===//
// IFRT --> VIFRT operations
//===----------------------------------------------------------------------===//

template <typename VifrtOpTy>
mlir::LogicalResult removeDefaultAttrs(
    const mlir::OpConversionPattern<VifrtOpTy>& pattern, VifrtOpTy vifrt_op,
    llvm::SmallVector<mlir::NamedAttribute>& vifrt_attrs) {
  absl::flat_hash_set<std::string> to_remove_attrs;
  if constexpr (std::is_same<VifrtOpTy, FuncOpV1>::value) {
    if (auto attr =
            llvm::dyn_cast<mlir::StringAttr>(vifrt_op.getSymVisibility());
        !attr || attr.str() == kVifrtDefaultString) {
      to_remove_attrs.insert("sym_visibility");
    }
    if (auto attr = llvm::dyn_cast<mlir::ArrayAttr>(vifrt_op.getArgAttrs());
        !attr || attr.size() == 0) {
      to_remove_attrs.insert("arg_attrs");
      ;
    }
    if (auto attr = llvm::dyn_cast<mlir::ArrayAttr>(vifrt_op.getResAttrs());
        !attr || attr.size() == 0) {
      to_remove_attrs.insert("res_attrs");
    }
  }
  llvm::erase_if(vifrt_attrs, [&](mlir::NamedAttribute attr) {
    return to_remove_attrs.contains(attr.getName().str());
  });
  return mlir::success();
}

// Returns a `SymbolRefAttr` for the given `CallOpV1`.
// The call op contains the symbol as a string, and this method converts it
// into a `SymbolRefAttr`. It handles the case where the symbol is a nested
// symbol, e.g., `@foo::@bar::@baz`.
mlir::FailureOr<mlir::SymbolRefAttr> getCalleeSymbolRef(CallOpV1 call_op) {
  mlir::StringAttr callee_symbol_ref_str_attr =
      llvm::dyn_cast_or_null<mlir::StringAttr>(call_op.getCalleeAttr());
  if (!callee_symbol_ref_str_attr) {
    return mlir::failure();
  }
  // It is important to call `getValue()` on the `StringAttr` to get the
  // unescaped string instead of the escaped string.
  std::vector<std::string> symbol_strs = absl::StrSplit(
      callee_symbol_ref_str_attr.getValue().str(), absl::ByString("::@"));
  if (symbol_strs.empty()) {
    return mlir::failure();
  }
  mlir::StringAttr root_symbol_ref;
  llvm::SmallVector<mlir::FlatSymbolRefAttr> symbol_ref_attrs;
  for (const auto [idx, symbol_str] : llvm::enumerate(symbol_strs)) {
    if (idx == 0) {
      if (symbol_str.empty() || symbol_str[0] != '@') {
        return mlir::failure();
      }
      symbol_str.erase(0, 1);
      root_symbol_ref = mlir::StringAttr::get(call_op.getContext(), symbol_str);
    } else {
      symbol_ref_attrs.push_back(
          mlir::FlatSymbolRefAttr::get(call_op.getContext(), symbol_str));
    }
  }
  return mlir::SymbolRefAttr::get(root_symbol_ref, symbol_ref_attrs);
}

// Generic op conversions. VIFRT has a 1:1 mapping for all ops in IFRT, and
// mlir::func::FuncOp, mlir::func::CallOp, and mlir::func::ReturnOp.
template <typename VifrtOpTy>
class VifrtToIfrtOpConverter : public mlir::OpConversionPattern<VifrtOpTy> {
 public:
  using mlir::OpConversionPattern<VifrtOpTy>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      VifrtOpTy vifrt_op, typename VifrtOpTy::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const final {
    // Convert the VIFRT result types to IFRT types.
    llvm::SmallVector<mlir::Type> ifrt_types;
    if (mlir::failed(this->getTypeConverter()->convertTypes(
            vifrt_op->getResultTypes(), ifrt_types))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed VIFRT to IFRT type conversion\n");
      return mlir::failure();
    }

    // Convert the IFRT attributes to VIFRT attributes.
    llvm::SmallVector<mlir::NamedAttribute> ifrt_attrs;
    llvm::DenseSet<mlir::StringAttr> already_converted_attrs;
    // Special case operations.
    if constexpr (std::is_same<VifrtOpTy, CallOpV1>::value) {
      auto call_op = static_cast<CallOpV1>(vifrt_op);
      auto callee_symbol_ref = getCalleeSymbolRef(call_op);
      if (mlir::failed(callee_symbol_ref)) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to get callee symbol ref from "
                                << call_op << '\n');
        return mlir::failure();
      }
      ifrt_attrs.push_back(
          {call_op.getCalleeAttrName(), callee_symbol_ref.value()});
      already_converted_attrs.insert(call_op.getCalleeAttrName());
    }

    llvm::SmallVector<mlir::NamedAttribute> vifrt_attrs =
        llvm::to_vector(vifrt_op->getAttrs());
    if (mlir::failed(removeDefaultAttrs(*this, vifrt_op, vifrt_attrs))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to remove default VIFRT attributes\n");
      return mlir::failure();
    }
    for (mlir::NamedAttribute vifrt_attr : vifrt_attrs) {
      // Skip special case attributes, which have already been converted.
      if (already_converted_attrs.contains(vifrt_attr.getName())) {
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "Converting " << vifrt_attr.getName() << ", "
                              << vifrt_attr.getValue() << '\n');
      auto attr_name = getAttrNameFromVifrtToIfrt(vifrt_attr);
      if (mlir::failed(attr_name)) {
        LLVM_DEBUG(llvm::dbgs() << "Failed to convert: " << vifrt_attr.getName()
                                << ", " << vifrt_attr.getValue() << '\n');
        return mlir::failure();
      }
      if (auto ifrt_attr =
              convertGeneric(vifrt_attr.getValue(), this->getTypeConverter())) {
        ifrt_attrs.push_back({attr_name.value(), ifrt_attr});
      } else {
        return mlir::failure();
      }
    }

    // The operands have already been converted to IFRT by the dialect
    // conversion infrastructure.
    mlir::ValueRange ifrt_operands = adaptor.getOperands();

    // Convert the IFRT op to a VIFRT equivalent op.
    VifrtToIfrtOp<VifrtOpTy> ifrt_op =
        rewriter.create<VifrtToIfrtOp<VifrtOpTy>>(vifrt_op.getLoc(), ifrt_types,
                                                  ifrt_operands, ifrt_attrs);

    // Convert the VIFRT region types to IFRT region types.
    for (auto [vifrt_region, ifrt_region] :
         llvm::zip(vifrt_op->getRegions(), ifrt_op->getRegions())) {
      rewriter.inlineRegionBefore(vifrt_region, ifrt_region, ifrt_region.end());
      if (mlir::failed(rewriter.convertRegionTypes(
              &ifrt_region, *this->getTypeConverter(),
              /*entryConversion=*/nullptr)))
        return mlir::failure();
    }
    rewriter.replaceOp(vifrt_op, ifrt_op);
    return mlir::success();
  }
};

template <typename... IfrtOpTypes>
void populateVifrtToIfrtPatterns(mlir::RewritePatternSet* patterns,
                                 mlir::TypeConverter* converter,
                                 mlir::MLIRContext* context) {
  patterns->add<VifrtToIfrtOpConverter<IfrtToVifrtOp<IfrtOpTypes>>...>(
      *converter, context);
}

// Verifies if a given type or attribute is from the IFRT dialect.
template <typename TypeOrAttr>
bool isFromIfrt(TypeOrAttr t) {
  return t.getDialect().getNamespace() == IfrtDialect::getDialectNamespace();
}

template <typename TypeOrAttr>
bool allFromIfrt(llvm::ArrayRef<TypeOrAttr> range) {
  return llvm::all_of(range, isFromIfrt<TypeOrAttr>);
}

}  // namespace

struct VifrtLegalizeToIfrtPass
    : public impl::VifrtLegalizeToIfrtPassBase<VifrtLegalizeToIfrtPass> {
  mlir::LogicalResult initialize(mlir::MLIRContext* context) override {
    target = std::make_shared<mlir::ConversionTarget>(*context);
    target->addIllegalDialect<VifrtDialect>();
    target->addLegalDialect<IfrtDialect>();
    // FuncDialect is used for atom programs and IFRT functions.
    target->addLegalDialect<mlir::func::FuncDialect>();
    mlir::RewritePatternSet patterns_(context);
    // Populate the patterns for the generic IFRT op to VIFRT op conversions.
    populateVifrtToIfrtPatterns(&patterns_, &converter, context);
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
  VifrtToIfrtTypeConverter converter;
  mlir::FrozenRewritePatternSet patterns;
  std::shared_ptr<mlir::ConversionTarget> target;
};

void populateVifrtToIfrtPatterns(mlir::RewritePatternSet* patterns,
                                 mlir::TypeConverter* converter,
                                 mlir::MLIRContext* context) {
  populateVifrtToIfrtPatterns<
#define GET_OP_LIST
#include "xla/python/ifrt/ir/ifrt_ops.cc.inc"
      , mlir::func::CallOp, mlir::func::FuncOp, mlir::func::ReturnOp>(
      patterns, converter, context);
}

}  // namespace ifrt
}  // namespace xla
