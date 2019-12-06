//===- ConvertStandardToSPIRV.h - Convert to SPIR-V dialect -----*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Provides type converters and patterns to convert from standard types/ops to
// SPIR-V types and operations. Also provides utilities and base classes to use
// while targeting SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOSPIRV_CONVERTSTANDARDTOSPIRV_H
#define MLIR_CONVERSION_STANDARDTOSPIRV_CONVERTSTANDARDTOSPIRV_H

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Support/StringExtras.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class LoadOp;
class ReturnOp;
class StoreOp;

/// Type conversion from Standard Types to SPIR-V Types.
class SPIRVBasicTypeConverter : public TypeConverter {
public:
  /// Converts types to SPIR-V supported types.
  virtual Type convertType(Type t);
};

/// Converts a function type according to the requirements of a SPIR-V entry
/// function. The arguments need to be converted to spv.Variables of spv.ptr
/// types so that they could be bound by the runtime.
class SPIRVTypeConverter final : public TypeConverter {
public:
  explicit SPIRVTypeConverter(SPIRVBasicTypeConverter *basicTypeConverter)
      : basicTypeConverter(basicTypeConverter) {}

  /// Converts types to SPIR-V types using the basic type converter.
  Type convertType(Type t) override;

  /// Gets the basic type converter.
  SPIRVBasicTypeConverter *getBasicTypeConverter() const {
    return basicTypeConverter;
  }

private:
  SPIRVBasicTypeConverter *basicTypeConverter;
};

/// Base class to define a conversion pattern to translate Ops into SPIR-V.
template <typename OpTy> class SPIRVOpLowering : public ConversionPattern {
public:
  SPIRVOpLowering(MLIRContext *context, SPIRVTypeConverter &typeConverter)
      : ConversionPattern(OpTy::getOperationName(), 1, context),
        typeConverter(typeConverter) {}

protected:
  /// Gets the global variable associated with a builtin and add
  /// it if it doesnt exist.
  Value *loadFromBuiltinVariable(Operation *op, spirv::BuiltIn builtin,
                                 ConversionPatternRewriter &rewriter) const {
    auto moduleOp = op->getParentOfType<spirv::ModuleOp>();
    if (!moduleOp) {
      op->emitError("expected operation to be within a SPIR-V module");
      return nullptr;
    }
    auto varOp =
        getOrInsertBuiltinVariable(moduleOp, op->getLoc(), builtin, rewriter);
    auto ptr = rewriter
                   .create<spirv::AddressOfOp>(op->getLoc(), varOp.type(),
                                               rewriter.getSymbolRefAttr(varOp))
                   .pointer();
    return rewriter.create<spirv::LoadOp>(
        op->getLoc(),
        ptr->getType().template cast<spirv::PointerType>().getPointeeType(),
        ptr, /*memory_access =*/nullptr, /*alignment =*/nullptr);
  }

  /// Type lowering class.
  SPIRVTypeConverter &typeConverter;

private:
  /// Look through all global variables in `moduleOp` and check if there is a
  /// spv.globalVariable that has the same `builtin` attribute.
  spirv::GlobalVariableOp getBuiltinVariable(spirv::ModuleOp &moduleOp,
                                             spirv::BuiltIn builtin) const {
    for (auto varOp : moduleOp.getBlock().getOps<spirv::GlobalVariableOp>()) {
      if (auto builtinAttr = varOp.getAttrOfType<StringAttr>(convertToSnakeCase(
              stringifyDecoration(spirv::Decoration::BuiltIn)))) {
        auto varBuiltIn = spirv::symbolizeBuiltIn(builtinAttr.getValue());
        if (varBuiltIn && varBuiltIn.getValue() == builtin) {
          return varOp;
        }
      }
    }
    return nullptr;
  }

  /// Gets name of global variable for a buitlin.
  std::string getBuiltinVarName(spirv::BuiltIn builtin) const {
    return std::string("__builtin_var_") + stringifyBuiltIn(builtin).str() +
           "__";
  }

  /// Gets or inserts a global variable for a builtin within a module.
  spirv::GlobalVariableOp
  getOrInsertBuiltinVariable(spirv::ModuleOp &moduleOp, Location loc,
                             spirv::BuiltIn builtin,
                             ConversionPatternRewriter &builder) const {
    if (auto varOp = getBuiltinVariable(moduleOp, builtin)) {
      return varOp;
    }
    auto ip = builder.saveInsertionPoint();
    builder.setInsertionPointToStart(&moduleOp.getBlock());
    auto name = getBuiltinVarName(builtin);
    spirv::GlobalVariableOp newVarOp;
    switch (builtin) {
    case spirv::BuiltIn::NumWorkgroups:
    case spirv::BuiltIn::WorkgroupSize:
    case spirv::BuiltIn::WorkgroupId:
    case spirv::BuiltIn::LocalInvocationId:
    case spirv::BuiltIn::GlobalInvocationId: {
      auto ptrType = spirv::PointerType::get(
          builder.getVectorType({3}, builder.getIntegerType(32)),
          spirv::StorageClass::Input);
      newVarOp = builder.create<spirv::GlobalVariableOp>(
          loc, builder.getTypeAttr(ptrType), builder.getStringAttr(name),
          nullptr);
      newVarOp.setAttr(
          convertToSnakeCase(stringifyDecoration(spirv::Decoration::BuiltIn)),
          builder.getStringAttr(stringifyBuiltIn(builtin)));
      break;
    }
    default:
      emitError(loc, "unimplemented builtin variable generation for ")
          << stringifyBuiltIn(builtin);
    }
    builder.restoreInsertionPoint(ip);
    return newVarOp;
  }
};

/// Legalizes a function as a non-entry function.
LogicalResult lowerFunction(FuncOp funcOp, SPIRVTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter,
                            FuncOp &newFuncOp);

/// Legalizes a function as an entry function.
LogicalResult lowerAsEntryFunction(FuncOp funcOp,
                                   SPIRVTypeConverter *typeConverter,
                                   ConversionPatternRewriter &rewriter,
                                   FuncOp &newFuncOp);

/// Finalizes entry function legalization. Inserts the spv.EntryPoint and
/// spv.ExecutionMode ops.
LogicalResult finalizeEntryFunction(FuncOp newFuncOp, OpBuilder &builder);

/// Appends to a pattern list additional patterns for translating StandardOps to
/// SPIR-V ops.
void populateStandardToSPIRVPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOSPIRV_CONVERTSTANDARDTOSPIRV_H
