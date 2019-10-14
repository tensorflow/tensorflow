//===- DecorateSPIRVCompositeTypeLayoutPass.cpp - Decorate composite type -===//
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
// This file implements a pass to decorate the composite types used by
// composite objects in the StorageBuffer, PhysicalStorageBuffer, Uniform, and
// PushConstant storage classes with layout information. See SPIR-V spec
// "2.16.2. Validation Rules for Shader Capabilities" for more details.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

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
namespace {
class VulkanLayoutUtils {
public:
  using Alignment = uint64_t;

  /// Returns a new type with layout info. Assigns the type size in bytes to the
  /// `size`. Assigns the type alignment in bytes to the `alignment`.
  static Type decorateType(spirv::StructType structType,
                           spirv::StructType::LayoutInfo &size,
                           Alignment &alignment);
  /// Checks whether a type is legal in terms of Vulkan layout info
  /// decoration. A type is dynamically illegal if it's a composite type in the
  /// StorageBuffer, PhysicalStorageBuffer, Uniform, and PushConstant Storage
  /// Classes without layout informtation.
  static bool isLegalType(Type type);

private:
  static Type decorateType(Type type, spirv::StructType::LayoutInfo &size,
                           Alignment &alignment);
  static Type decorateType(VectorType vectorType,
                           spirv::StructType::LayoutInfo &size,
                           Alignment &alignment);
  static Type decorateType(spirv::ArrayType arrayType,
                           spirv::StructType::LayoutInfo &size,
                           Alignment &alignment);
  /// Calculates the alignment for the given scalar type.
  static Alignment getScalarTypeAlignment(Type scalarType);
};

Type VulkanLayoutUtils::decorateType(spirv::StructType structType,
                                     spirv::StructType::LayoutInfo &size,
                                     VulkanLayoutUtils::Alignment &alignment) {
  if (structType.getNumElements() == 0) {
    return structType;
  }

  llvm::SmallVector<Type, 4> memberTypes;
  llvm::SmallVector<spirv::StructType::LayoutInfo, 4> layoutInfo;
  llvm::SmallVector<spirv::StructType::MemberDecorationInfo, 4>
      memberDecorations;

  spirv::StructType::LayoutInfo structMemberOffset = 0;
  VulkanLayoutUtils::Alignment maxMemberAlignment = 1;

  for (uint32_t i = 0, e = structType.getNumElements(); i < e; ++i) {
    spirv::StructType::LayoutInfo memberSize = 0;
    VulkanLayoutUtils::Alignment memberAlignment = 1;

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

Type VulkanLayoutUtils::decorateType(Type type,
                                     spirv::StructType::LayoutInfo &size,
                                     VulkanLayoutUtils::Alignment &alignment) {
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
                                     spirv::StructType::LayoutInfo &size,
                                     VulkanLayoutUtils::Alignment &alignment) {
  const auto numElements = vectorType.getNumElements();
  auto elementType = vectorType.getElementType();
  spirv::StructType::LayoutInfo elementSize = 0;
  VulkanLayoutUtils::Alignment elementAlignment = 1;

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
                                     spirv::StructType::LayoutInfo &size,
                                     VulkanLayoutUtils::Alignment &alignment) {
  const auto numElements = arrayType.getNumElements();
  auto elementType = arrayType.getElementType();
  spirv::ArrayType::LayoutInfo elementSize = 0;
  VulkanLayoutUtils::Alignment elementAlignment = 1;

  auto memberType = VulkanLayoutUtils::decorateType(elementType, elementSize,
                                                    elementAlignment);
  // According to the Vulkan spec:
  // "An array has a base alignment equal to the base alignment of its element
  // type."
  size = elementSize * numElements;
  alignment = elementAlignment;
  return spirv::ArrayType::get(memberType, numElements, elementSize);
}

VulkanLayoutUtils::Alignment
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
} // namespace

namespace {
class SPIRVGlobalVariableOpLayoutInfoDecoration
    : public OpRewritePattern<spirv::GlobalVariableOp> {
public:
  using OpRewritePattern<spirv::GlobalVariableOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(spirv::GlobalVariableOp op,
                                     PatternRewriter &rewriter) const override {
    spirv::StructType::LayoutInfo structSize = 0;
    VulkanLayoutUtils::Alignment structAlignment = 1;
    SmallVector<NamedAttribute, 4> globalVarAttrs;

    auto ptrType = op.type().cast<spirv::PointerType>();
    auto structType = VulkanLayoutUtils::decorateType(
        ptrType.getPointeeType().cast<spirv::StructType>(), structSize,
        structAlignment);
    auto decoratedType =
        spirv::PointerType::get(structType, ptrType.getStorageClass());

    // Save all named attributes except "type" attribute.
    for (const auto &attr : op.getAttrs()) {
      if (attr.first == "type") {
        continue;
      }
      globalVarAttrs.push_back(attr);
    }

    rewriter.replaceOpWithNewOp<spirv::GlobalVariableOp>(
        op, rewriter.getTypeAttr(decoratedType), globalVarAttrs);
    return matchSuccess();
  }
};

class SPIRVAddressOfOpLayoutInfoDecoration
    : public OpRewritePattern<spirv::AddressOfOp> {
public:
  using OpRewritePattern<spirv::AddressOfOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(spirv::AddressOfOp op,
                                     PatternRewriter &rewriter) const override {
    auto spirvModule = op.getParentOfType<spirv::ModuleOp>();
    auto varName = op.variable();
    auto varOp = spirvModule.lookupSymbol<spirv::GlobalVariableOp>(varName);

    rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(
        op, varOp.type(), rewriter.getSymbolRefAttr(varName));
    return matchSuccess();
  }
};
} // namespace

static void populateSPIRVLayoutInfoPatterns(OwningRewritePatternList &patterns,
                                            MLIRContext *ctx) {
  patterns.insert<SPIRVGlobalVariableOpLayoutInfoDecoration,
                  SPIRVAddressOfOpLayoutInfoDecoration>(ctx);
}

namespace {
class DecorateSPIRVCompositeTypeLayoutPass
    : public ModulePass<DecorateSPIRVCompositeTypeLayoutPass> {
private:
  void runOnModule() override;
};

void DecorateSPIRVCompositeTypeLayoutPass::runOnModule() {
  auto module = getModule();
  OwningRewritePatternList patterns;
  populateSPIRVLayoutInfoPatterns(patterns, module.getContext());
  ConversionTarget target(*(module.getContext()));
  target.addLegalDialect<spirv::SPIRVDialect>();
  target.addLegalOp<FuncOp>();
  target.addDynamicallyLegalOp<spirv::GlobalVariableOp>(
      [](spirv::GlobalVariableOp op) {
        return VulkanLayoutUtils::isLegalType(op.type());
      });

  // Change the type for the direct users.
  target.addDynamicallyLegalOp<spirv::AddressOfOp>([](spirv::AddressOfOp op) {
    return VulkanLayoutUtils::isLegalType(op.pointer()->getType());
  });

  // TODO: Change the type for the indirect users such as spv.Load, spv.Store,
  // spv.FunctionCall and so on.

  for (auto spirvModule : module.getOps<spirv::ModuleOp>()) {
    if (failed(applyFullConversion(spirvModule, target, patterns))) {
      signalPassFailure();
    }
  }
}
} // namespace

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::spirv::createDecorateSPIRVCompositeTypeLayoutPass() {
  return std::make_unique<DecorateSPIRVCompositeTypeLayoutPass>();
}

static PassRegistration<DecorateSPIRVCompositeTypeLayoutPass>
    pass("decorate-spirv-composite-type-layout",
         "Decorate SPIR-V composite type with layout info");
