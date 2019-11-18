//===- SPIRVLowering.h - SPIR-V lowering utilities  -------------*- C++ -*-===//
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
// Defines, utilities and base classes to use while targeting SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_SPIRVLOWERING_H
#define MLIR_DIALECT_SPIRV_SPIRVLOWERING_H

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Support/StringExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {

/// Type conversion from Standard Types to SPIR-V Types.
class SPIRVBasicTypeConverter : public TypeConverter {
public:
  /// Converts types to SPIR-V supported types.
  virtual Type convertType(Type t);
};

/// Converts a function type according to the requirements of a SPIR-V entry
/// function. The arguments need to be converted to spv.GlobalVariables of
/// spv.ptr types so that they could be bound by the runtime.
class SPIRVTypeConverter final : public TypeConverter {
public:
  explicit SPIRVTypeConverter(SPIRVBasicTypeConverter *basicTypeConverter)
      : basicTypeConverter(basicTypeConverter) {}

  /// Converts types to SPIR-V types using the basic type converter.
  Type convertType(Type t) override;

  /// Gets the basic type converter.
  Type convertBasicType(Type t) { return basicTypeConverter->convertType(t); }

private:
  SPIRVBasicTypeConverter *basicTypeConverter;
};

/// Base class to define a conversion pattern to translate Ops into SPIR-V.
template <typename SourceOp>
class SPIRVOpLowering : public OpConversionPattern<SourceOp> {
public:
  SPIRVOpLowering(MLIRContext *context, SPIRVTypeConverter &typeConverter,
                  PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(context, benefit),
        typeConverter(typeConverter) {}

protected:
  /// Type lowering class.
  SPIRVTypeConverter &typeConverter;

private:
};

namespace spirv {
/// Returns a value that represents a builtin variable value within the SPIR-V
/// module.
Value *getBuiltinVariableValue(Operation *op, spirv::BuiltIn builtin,
                               OpBuilder &builder);

/// Legalizes a function as an entry function.
LogicalResult lowerAsEntryFunction(FuncOp funcOp,
                                   SPIRVTypeConverter *typeConverter,
                                   ConversionPatternRewriter &rewriter,
                                   FuncOp &newFuncOp);

/// Finalizes entry function legalization. Inserts the spv.EntryPoint and
/// spv.ExecutionMode ops.
LogicalResult finalizeEntryFunction(FuncOp newFuncOp, OpBuilder &builder);

} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_SPIRVLOWERING_H
