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
class SPIRVTypeConverter : public TypeConverter {
public:
  explicit SPIRVTypeConverter(MLIRContext *context);

  /// Converts types to SPIR-V supported types.
  Type convertType(Type t) override;

protected:
  spirv::SPIRVDialect *spirvDialect;
};

/// Converts a function type according to the requirements of a SPIR-V entry
/// function. The arguments need to be converted to spv.Variables of spv.ptr
/// types so that they could be bound by the runtime.
class SPIRVEntryFnTypeConverter final : public SPIRVTypeConverter {
public:
  using SPIRVTypeConverter::SPIRVTypeConverter;

  /// Method to convert argument of a function. The `type` is converted to
  /// spv.ptr<type, Uniform>.
  // TODO(ravishankarm) : Support other storage classes.
  LogicalResult convertSignatureArg(unsigned inputNo, Type type,
                                    SignatureConversion &result) override;
};

/// Base class to define a conversion pattern to translate Ops into SPIR-V.
template <typename OpTy> class SPIRVOpLowering : public ConversionPattern {
public:
  SPIRVOpLowering(MLIRContext *context, SPIRVTypeConverter &typeConverter,
                  SPIRVEntryFnTypeConverter &entryFnConverter)
      : ConversionPattern(OpTy::getOperationName(), 1, context),
        typeConverter(typeConverter), entryFnConverter(entryFnConverter) {}

protected:
  // Type lowering class.
  SPIRVTypeConverter &typeConverter;

  // Entry function signature converter.
  SPIRVEntryFnTypeConverter &entryFnConverter;
};

/// Base Class for legalize a FuncOp within a spv.module. This class can be
/// extended to implement a ConversionPattern to lower a FuncOp. It provides
/// hooks to legalize a FuncOp as a simple function, or as an entry function.
class SPIRVFnLowering : public SPIRVOpLowering<FuncOp> {
public:
  using SPIRVOpLowering<FuncOp>::SPIRVOpLowering;

protected:
  /// Method to legalize the function as a non-entry function.
  LogicalResult lowerFunction(FuncOp funcOp, ArrayRef<Value *> operands,
                              ConversionPatternRewriter &rewriter,
                              FuncOp &newFuncOp) const;

  /// Method to legalize the function as an entry function.
  LogicalResult lowerAsEntryFunction(FuncOp funcOp, ArrayRef<Value *> operands,
                                     ConversionPatternRewriter &rewriter,
                                     FuncOp &newFuncOp) const;
};

/// Appends to a pattern list additional patterns for translating StandardOps to
/// SPIR-V ops.
void populateStandardToSPIRVPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOSPIRV_CONVERTSTANDARDTOSPIRV_H
