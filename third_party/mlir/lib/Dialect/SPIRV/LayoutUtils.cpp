//===-- LayoutUtils.cpp - Decorate composite type with layout information -===//
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
// This file implements Utilities used to get alignment and layout information
// for types in SPIR-V dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/SPIRV/LayoutUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"

using namespace mlir;

spirv::StructType
VulkanLayoutUtils::decorateType(spirv::StructType structType,
                                VulkanLayoutUtils::Size &size,
                                VulkanLayoutUtils::Size &alignment) {
  if (structType.getNumElements() == 0) {
    return structType;
  }

  SmallVector<Type, 4> memberTypes;
  SmallVector<VulkanLayoutUtils::Size, 4> layoutInfo;
  SmallVector<spirv::StructType::MemberDecorationInfo, 4> memberDecorations;

  VulkanLayoutUtils::Size structMemberOffset = 0;
  VulkanLayoutUtils::Size maxMemberAlignment = 1;

  for (uint32_t i = 0, e = structType.getNumElements(); i < e; ++i) {
    VulkanLayoutUtils::Size memberSize = 0;
    VulkanLayoutUtils::Size memberAlignment = 1;

    auto memberType = VulkanLayoutUtils::decorateType(
        structType.getElementType(i), memberSize, memberAlignment);
    structMemberOffset = llvm::alignTo(structMemberOffset, memberAlignment);
    memberTypes.push_back(memberType);
    layoutInfo.push_back(structMemberOffset);
    // According to the Vulkan spec:
    // "A structure has a base alignment equal to the largest base alignment of
    // any of its members."
    structMemberOffset += memberSize;
    maxMemberAlignment = std::max(maxMemberAlignment, memberAlignment);
  }

  // According to the Vulkan spec:
  // "The Offset decoration of a member must not place it between the end of a
  // structure or an array and the next multiple of the alignment of that
  // structure or array."
  size = llvm::alignTo(structMemberOffset, maxMemberAlignment);
  alignment = maxMemberAlignment;
  structType.getMemberDecorations(memberDecorations);
  return spirv::StructType::get(memberTypes, layoutInfo, memberDecorations);
}

Type VulkanLayoutUtils::decorateType(Type type, VulkanLayoutUtils::Size &size,
                                     VulkanLayoutUtils::Size &alignment) {
  if (spirv::SPIRVDialect::isValidScalarType(type)) {
    alignment = VulkanLayoutUtils::getScalarTypeAlignment(type);
    // Vulkan spec does not specify any padding for a scalar type.
    size = alignment;
    return type;
  }

  switch (type.getKind()) {
  case spirv::TypeKind::Struct:
    return VulkanLayoutUtils::decorateType(type.cast<spirv::StructType>(), size,
                                           alignment);
  case spirv::TypeKind::Array:
    return VulkanLayoutUtils::decorateType(type.cast<spirv::ArrayType>(), size,
                                           alignment);
  case StandardTypes::Vector:
    return VulkanLayoutUtils::decorateType(type.cast<VectorType>(), size,
                                           alignment);
  default:
    llvm_unreachable("unhandled SPIR-V type");
  }
}

Type VulkanLayoutUtils::decorateType(VectorType vectorType,
                                     VulkanLayoutUtils::Size &size,
                                     VulkanLayoutUtils::Size &alignment) {
  const auto numElements = vectorType.getNumElements();
  auto elementType = vectorType.getElementType();
  VulkanLayoutUtils::Size elementSize = 0;
  VulkanLayoutUtils::Size elementAlignment = 1;

  auto memberType = VulkanLayoutUtils::decorateType(elementType, elementSize,
                                                    elementAlignment);
  // According to the Vulkan spec:
  // 1. "A two-component vector has a base alignment equal to twice its scalar
  // alignment."
  // 2. "A three- or four-component vector has a base alignment equal to four
  // times its scalar alignment."
  size = elementSize * numElements;
  alignment = numElements == 2 ? elementAlignment * 2 : elementAlignment * 4;
  return VectorType::get(numElements, memberType);
}

Type VulkanLayoutUtils::decorateType(spirv::ArrayType arrayType,
                                     VulkanLayoutUtils::Size &size,
                                     VulkanLayoutUtils::Size &alignment) {
  const auto numElements = arrayType.getNumElements();
  auto elementType = arrayType.getElementType();
  spirv::ArrayType::LayoutInfo elementSize = 0;
  VulkanLayoutUtils::Size elementAlignment = 1;

  auto memberType = VulkanLayoutUtils::decorateType(elementType, elementSize,
                                                    elementAlignment);
  // According to the Vulkan spec:
  // "An array has a base alignment equal to the base alignment of its element
  // type."
  size = elementSize * numElements;
  alignment = elementAlignment;
  return spirv::ArrayType::get(memberType, numElements, elementSize);
}

VulkanLayoutUtils::Size
VulkanLayoutUtils::getScalarTypeAlignment(Type scalarType) {
  // According to the Vulkan spec:
  // 1. "A scalar of size N has a scalar alignment of N."
  // 2. "A scalar has a base alignment equal to its scalar alignment."
  // 3. "A scalar, vector or matrix type has an extended alignment equal to its
  // base alignment."
  auto bitWidth = scalarType.getIntOrFloatBitWidth();
  if (bitWidth == 1)
    return 1;
  return bitWidth / 8;
}

bool VulkanLayoutUtils::isLegalType(Type type) {
  auto ptrType = type.dyn_cast<spirv::PointerType>();
  if (!ptrType) {
    return true;
  }

  auto storageClass = ptrType.getStorageClass();
  auto structType = ptrType.getPointeeType().dyn_cast<spirv::StructType>();
  if (!structType) {
    return true;
  }

  switch (storageClass) {
  case spirv::StorageClass::Uniform:
  case spirv::StorageClass::StorageBuffer:
  case spirv::StorageClass::PushConstant:
  case spirv::StorageClass::PhysicalStorageBuffer:
    return structType.hasLayout() || !structType.getNumElements();
  default:
    return true;
  }
}
