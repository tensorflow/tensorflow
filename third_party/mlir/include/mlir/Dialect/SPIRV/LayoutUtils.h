//===-- LayoutUtils.h - Decorate composite type with layout information ---===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utilities used to get alignment and layout information for
// types in SPIR-V dialect.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_SPIRV_LAYOUTUTILS_H_
#define MLIR_DIALECT_SPIRV_LAYOUTUTILS_H_

#include <cstdint>

namespace mlir {
class Type;
class VectorType;
namespace spirv {
class StructType;
class ArrayType;
} // namespace spirv

/// According to the Vulkan spec "14.5.4. Offset and Stride Assignment":
/// "There are different alignment requirements depending on the specific
/// resources and on the features enabled on the device."
///
/// There are 3 types of alignment: scalar, base, extended.
/// See the spec for details.
///
/// Note: Even if scalar alignment is supported, it is generally more
/// performant to use the base alignment. So here the calculation is based on
/// base alignment.
///
/// The memory layout must obey the following rules:
/// 1. The Offset decoration of any member must be a multiple of its alignment.
/// 2. Any ArrayStride or MatrixStride decoration must be a multiple of the
/// alignment of the array or matrix as defined above.
///
/// According to the SPIR-V spec:
/// "The ArrayStride, MatrixStride, and Offset decorations must be large
/// enough to hold the size of the objects they affect (that is, specifying
/// overlap is invalid)."
class VulkanLayoutUtils {
public:
  using Size = uint64_t;

  /// Returns a new StructType with layout info. Assigns the type size in bytes
  /// to the `size`. Assigns the type alignment in bytes to the `alignment`.
  static spirv::StructType decorateType(spirv::StructType structType,
                                        Size &size, Size &alignment);
  /// Checks whether a type is legal in terms of Vulkan layout info
  /// decoration. A type is dynamically illegal if it's a composite type in the
  /// StorageBuffer, PhysicalStorageBuffer, Uniform, and PushConstant Storage
  /// Classes without layout information.
  static bool isLegalType(Type type);

private:
  static Type decorateType(Type type, Size &size, Size &alignment);
  static Type decorateType(VectorType vectorType, Size &size, Size &alignment);
  static Type decorateType(spirv::ArrayType arrayType, Size &size,
                           Size &alignment);
  /// Calculates the alignment for the given scalar type.
  static Size getScalarTypeAlignment(Type scalarType);
};

} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_LAYOUTUTILS_H_
