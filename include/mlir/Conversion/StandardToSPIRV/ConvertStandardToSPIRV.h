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

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

namespace spirv {
class SPIRVDialect;
}

/// Type conversion from Standard Types to SPIR-V Types.
class SPIRVBasicTypeConverter : public TypeConverter {
public:
  explicit SPIRVBasicTypeConverter(MLIRContext *context);

  /// Converts types to SPIR-V supported types.
  virtual Type convertType(Type t);

protected:
  spirv::SPIRVDialect *spirvDialect;
};

/// Converts a function type according to the requirements of a SPIR-V entry
/// function. The arguments need to be converted to spv.Variables of spv.ptr
/// types so that they could be bound by the runtime.
class SPIRVTypeConverter final : public TypeConverter {
public:
  explicit SPIRVTypeConverter(SPIRVBasicTypeConverter *basicTypeConverter)
      : basicTypeConverter(basicTypeConverter) {}

  /// Convert types to SPIR-V types using the basic type converter.
  Type convertType(Type t) override {
    return basicTypeConverter->convertType(t);
  }

  /// Method to convert argument of a function. The `type` is converted to
  /// spv.ptr<type, Uniform>.
  // TODO(ravishankarm) : Support other storage classes.
  LogicalResult convertSignatureArg(unsigned inputNo, Type type,
                                    SignatureConversion &result) override;

  /// Get the basic type converter.
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
  // Type lowering class.
  SPIRVTypeConverter &typeConverter;
};

/// Method to legalize a function as a non-entry function.
LogicalResult lowerFunction(FuncOp funcOp, ArrayRef<Value *> operands,
                            SPIRVTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter,
                            FuncOp &newFuncOp);

/// Method to legalize a function as an entry function.
LogicalResult lowerAsEntryFunction(FuncOp funcOp, ArrayRef<Value *> operands,
                                   SPIRVTypeConverter *typeConverter,
                                   ConversionPatternRewriter &rewriter,
                                   FuncOp &newFuncOp);

/// Appends to a pattern list additional patterns for translating StandardOps to
/// SPIR-V ops.
void populateStandardToSPIRVPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOSPIRV_CONVERTSTANDARDTOSPIRV_H
