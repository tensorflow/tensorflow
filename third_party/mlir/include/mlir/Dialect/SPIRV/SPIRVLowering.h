//===- SPIRVLowering.h - SPIR-V lowering utilities  -------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines utilities to use while targeting SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_SPIRVLOWERING_H
#define MLIR_DIALECT_SPIRV_SPIRVLOWERING_H

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/StringExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {

/// Type conversion from standard types to SPIR-V types for shader interface.
///
/// For composite types, this converter additionally performs type wrapping to
/// satisfy shader interface requirements: shader interface types must be
/// pointers to structs.
class SPIRVTypeConverter final : public TypeConverter {
public:
  using TypeConverter::TypeConverter;

  /// Converts the given standard `type` to SPIR-V correspondence.
  Type convertType(Type type) override;

  /// Gets the SPIR-V correspondence for the standard index type.
  static Type getIndexType(MLIRContext *context);
};

/// Base class to define a conversion pattern to lower `SourceOp` into SPIR-V.
template <typename SourceOp>
class SPIRVOpLowering : public OpConversionPattern<SourceOp> {
public:
  SPIRVOpLowering(MLIRContext *context, SPIRVTypeConverter &typeConverter,
                  PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(context, benefit),
        typeConverter(typeConverter) {}

protected:
  SPIRVTypeConverter &typeConverter;
};

#include "mlir/Dialect/SPIRV/SPIRVLowering.h.inc"

namespace spirv {
/// Returns a value that represents a builtin variable value within the SPIR-V
/// module.
Value getBuiltinVariableValue(Operation *op, spirv::BuiltIn builtin,
                              OpBuilder &builder);

/// Attribute name for specifying argument ABI information.
StringRef getInterfaceVarABIAttrName();

/// Get the InterfaceVarABIAttr given its fields.
InterfaceVarABIAttr getInterfaceVarABIAttr(unsigned descriptorSet,
                                           unsigned binding,
                                           spirv::StorageClass storageClass,
                                           MLIRContext *context);

/// Attribute name for specifying entry point information.
StringRef getEntryPointABIAttrName();

/// Get the EntryPointABIAttr given its fields.
EntryPointABIAttr getEntryPointABIAttr(ArrayRef<int32_t> localSize,
                                       MLIRContext *context);

/// Sets the InterfaceVarABIAttr and EntryPointABIAttr for a function and its
/// arguments
LogicalResult setABIAttrs(FuncOp funcOp,
                          spirv::EntryPointABIAttr entryPointInfo,
                          ArrayRef<spirv::InterfaceVarABIAttr> argABIInfo);

} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_SPIRVLOWERING_H
