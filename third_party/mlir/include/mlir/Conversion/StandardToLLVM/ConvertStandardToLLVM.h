//===- ConvertStandardToLLVM.h - Convert to the LLVM dialect ----*- C++ -*-===//
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
// Provides a dialect conversion targeting the LLVM IR dialect.  By default, it
// converts Standard ops and types and provides hooks for dialect-specific
// extensions to the conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVM_H
#define MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVM_H

#include "mlir/Transforms/DialectConversion.h"

namespace llvm {
class IntegerType;
class LLVMContext;
class Module;
class Type;
} // namespace llvm

namespace mlir {

class UnrankedMemRefType;

namespace LLVM {
class LLVMDialect;
class LLVMType;
} // namespace LLVM

/// Conversion from types in the Standard dialect to the LLVM IR dialect.
class LLVMTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  LLVMTypeConverter(MLIRContext *ctx);

  /// Convert types to LLVM IR.  This calls `convertAdditionalType` to convert
  /// non-standard or non-builtin types.
  Type convertType(Type t) override;

  /// Convert a function type.  The arguments and results are converted one by
  /// one and results are packed into a wrapped LLVM IR structure type. `result`
  /// is populated with argument mapping.
  LLVM::LLVMType convertFunctionSignature(FunctionType type, bool isVariadic,
                                          SignatureConversion &result);

  /// Convert a non-empty list of types to be returned from a function into a
  /// supported LLVM IR type.  In particular, if more than one values is
  /// returned, create an LLVM IR structure type with elements that correspond
  /// to each of the MLIR types converted with `convertType`.
  Type packFunctionResults(ArrayRef<Type> types);

  /// Returns the LLVM context.
  llvm::LLVMContext &getLLVMContext();

  /// Returns the LLVM dialect.
  LLVM::LLVMDialect *getDialect() { return llvmDialect; }

  /// Promote the LLVM struct representation of all MemRef descriptors to stack
  /// and use pointers to struct to avoid the complexity of the
  /// platform-specific C/C++ ABI lowering related to struct argument passing.
  SmallVector<Value *, 4> promoteMemRefDescriptors(Location loc,
                                                   ArrayRef<Value *> opOperands,
                                                   ArrayRef<Value *> operands,
                                                   OpBuilder &builder);

  /// Promote the LLVM struct representation of one MemRef descriptor to stack
  /// and use pointer to struct to avoid the complexity of the platform-specific
  /// C/C++ ABI lowering related to struct argument passing.
  Value *promoteOneMemRefDescriptor(Location loc, Value *operand,
                                    OpBuilder &builder);

protected:
  /// LLVM IR module used to parse/create types.
  llvm::Module *module;
  LLVM::LLVMDialect *llvmDialect;

private:
  Type convertStandardType(Type type);

  // Convert a function type.  The arguments and results are converted one by
  // one.  Additionally, if the function returns more than one value, pack the
  // results into an LLVM IR structure type so that the converted function type
  // returns at most one result.
  Type convertFunctionType(FunctionType type);

  // Convert the index type.  Uses llvmModule data layout to create an integer
  // of the pointer bitwidth.
  Type convertIndexType(IndexType type);

  // Convert an integer type `i*` to `!llvm<"i*">`.
  Type convertIntegerType(IntegerType type);

  // Convert a floating point type: `f16` to `!llvm.half`, `f32` to
  // `!llvm.float` and `f64` to `!llvm.double`.  `bf16` is not supported
  // by LLVM.
  Type convertFloatType(FloatType type);

  // Convert a memref type into an LLVM type that captures the relevant data.
  // For statically-shaped memrefs, the resulting type is a pointer to the
  // (converted) memref element type. For dynamically-shaped memrefs, the
  // resulting type is an LLVM structure type that contains:
  //   1. a pointer to the (converted) memref element type
  //   2. as many index types as memref has dynamic dimensions.
  Type convertMemRefType(MemRefType type);

  // Convert an unranked memref type to an LLVM type that captures the
  // runtime rank and a pointer to the static ranked memref desc
  Type convertUnrankedMemRefType(UnrankedMemRefType type);

  // Convert a 1D vector type into an LLVM vector type.
  Type convertVectorType(VectorType type);

  // Get the LLVM representation of the index type based on the bitwidth of the
  // pointer as defined by the data layout of the module.
  LLVM::LLVMType getIndexType();

  // Extract an LLVM IR dialect type.
  LLVM::LLVMType unwrap(Type type);
};

/// Helper class to produce LLVM dialect operations extracting or inserting
/// values to a struct.
class StructBuilder {
public:
  /// Construct a helper for the given value.
  explicit StructBuilder(Value *v);
  /// Builds IR creating an `undef` value of the descriptor type.
  static StructBuilder undef(OpBuilder &builder, Location loc,
                             Type descriptorType);

  /*implicit*/ operator Value *() { return value; }

protected:
  // LLVM value
  Value *value;
  // Cached struct type.
  Type structType;

protected:
  /// Builds IR to extract a value from the struct at position pos
  Value *extractPtr(OpBuilder &builder, Location loc, unsigned pos);
  /// Builds IR to set a value in the struct at position pos
  void setPtr(OpBuilder &builder, Location loc, unsigned pos, Value *ptr);
};
/// Helper class to produce LLVM dialect operations extracting or inserting
/// elements of a MemRef descriptor. Wraps a Value pointing to the descriptor.
/// The Value may be null, in which case none of the operations are valid.
class MemRefDescriptor : public StructBuilder {
public:
  /// Construct a helper for the given descriptor value.
  explicit MemRefDescriptor(Value *descriptor);
  /// Builds IR creating an `undef` value of the descriptor type.
  static MemRefDescriptor undef(OpBuilder &builder, Location loc,
                                Type descriptorType);
  /// Builds IR extracting the allocated pointer from the descriptor.
  Value *allocatedPtr(OpBuilder &builder, Location loc);
  /// Builds IR inserting the allocated pointer into the descriptor.
  void setAllocatedPtr(OpBuilder &builder, Location loc, Value *ptr);

  /// Builds IR extracting the aligned pointer from the descriptor.
  Value *alignedPtr(OpBuilder &builder, Location loc);

  /// Builds IR inserting the aligned pointer into the descriptor.
  void setAlignedPtr(OpBuilder &builder, Location loc, Value *ptr);

  /// Builds IR extracting the offset from the descriptor.
  Value *offset(OpBuilder &builder, Location loc);

  /// Builds IR inserting the offset into the descriptor.
  void setOffset(OpBuilder &builder, Location loc, Value *offset);

  /// Builds IR extracting the pos-th size from the descriptor.
  Value *size(OpBuilder &builder, Location loc, unsigned pos);

  /// Builds IR inserting the pos-th size into the descriptor
  void setSize(OpBuilder &builder, Location loc, unsigned pos, Value *size);

  /// Builds IR extracting the pos-th size from the descriptor.
  Value *stride(OpBuilder &builder, Location loc, unsigned pos);

  /// Builds IR inserting the pos-th stride into the descriptor
  void setStride(OpBuilder &builder, Location loc, unsigned pos, Value *stride);

  /// Returns the (LLVM) type this descriptor points to.
  LLVM::LLVMType getElementType();

private:
  // Cached index type.
  Type indexType;
};

class UnrankedMemRefDescriptor : public StructBuilder {
public:
  /// Construct a helper for the given descriptor value.
  explicit UnrankedMemRefDescriptor(Value *descriptor);
  /// Builds IR creating an `undef` value of the descriptor type.
  static UnrankedMemRefDescriptor undef(OpBuilder &builder, Location loc,
                                        Type descriptorType);

  /// Builds IR extracting the rank from the descriptor
  Value *rank(OpBuilder &builder, Location loc);
  /// Builds IR setting the rank in the descriptor
  void setRank(OpBuilder &builder, Location loc, Value *value);
  /// Builds IR extracting ranked memref descriptor ptr
  Value *memRefDescPtr(OpBuilder &builder, Location loc);
  /// Builds IR setting ranked memref descriptor ptr
  void setMemRefDescPtr(OpBuilder &builder, Location loc, Value *value);
};
/// Base class for operation conversions targeting the LLVM IR dialect. Provides
/// conversion patterns with an access to the containing LLVMLowering for the
/// purpose of type conversions.
class LLVMOpLowering : public ConversionPattern {
public:
  LLVMOpLowering(StringRef rootOpName, MLIRContext *context,
                 LLVMTypeConverter &lowering, PatternBenefit benefit = 1);

protected:
  // Back-reference to the lowering class, used to call type and function
  // conversions accounting for potential extensions.
  LLVMTypeConverter &lowering;
};

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOLLVM_CONVERTSTANDARDTOLLVM_H
