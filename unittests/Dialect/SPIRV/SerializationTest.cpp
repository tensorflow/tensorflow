//===- SerializationTest.cpp - SPIR-V Serialization Tests -----------------===//
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
// This file contains corner case tests for the SPIR-V serializer that are not
// covered by normal serialization and deserialization roundtripping.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/Dialect/SPIRV/SPIRVBinaryUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "gmock/gmock.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class SerializationTest : public ::testing::Test {
protected:
  SerializationTest() { createModuleOp(); }

  void createModuleOp() {
    Builder builder(&context);
    OperationState state(UnknownLoc::get(&context),
                         spirv::ModuleOp::getOperationName());
    state.addAttribute("addressing_model",
                       builder.getI32IntegerAttr(static_cast<uint32_t>(
                           spirv::AddressingModel::Logical)));
    state.addAttribute("memory_model",
                       builder.getI32IntegerAttr(
                           static_cast<uint32_t>(spirv::MemoryModel::GLSL450)));
    spirv::ModuleOp::build(&builder, state);
    module = cast<spirv::ModuleOp>(Operation::create(state));
  }

  Type getFloatStructType() {
    OpBuilder opBuilder(module.body());
    llvm::SmallVector<Type, 1> elementTypes{opBuilder.getF32Type()};
    llvm::SmallVector<spirv::StructType::LayoutInfo, 1> layoutInfo{0};
    auto structType = spirv::StructType::get(elementTypes, layoutInfo);
    return structType;
  }

  void addGlobalVar(Type type, llvm::StringRef name) {
    OpBuilder opBuilder(module.body());
    auto ptrType = spirv::PointerType::get(type, spirv::StorageClass::Uniform);
    opBuilder.create<spirv::GlobalVariableOp>(
        UnknownLoc::get(&context), TypeAttr::get(ptrType),
        opBuilder.getStringAttr(name), nullptr);
  }

  bool findInstruction(llvm::function_ref<bool(spirv::Opcode opcode,
                                               ArrayRef<uint32_t> operands)>
                           matchFn) {
    auto binarySize = binary.size();
    auto begin = binary.begin();
    auto currOffset = spirv::kHeaderWordCount;

    while (currOffset < binarySize) {
      auto wordCount = binary[currOffset] >> 16;
      if (!wordCount || (currOffset + wordCount > binarySize)) {
        return false;
      }
      spirv::Opcode opcode =
          static_cast<spirv::Opcode>(binary[currOffset] & 0xffff);

      if (matchFn(opcode,
                  llvm::ArrayRef<uint32_t>(begin + currOffset + 1,
                                           begin + currOffset + wordCount))) {
        return true;
      }
      currOffset += wordCount;
    }
    return false;
  }

protected:
  MLIRContext context;
  spirv::ModuleOp module;
  SmallVector<uint32_t, 0> binary;
};

//===----------------------------------------------------------------------===//
// Block decoration
//===----------------------------------------------------------------------===//

TEST_F(SerializationTest, BlockDecorationTest) {
  auto structType = getFloatStructType();
  addGlobalVar(structType, "var0");
  ASSERT_TRUE(succeeded(spirv::serialize(module, binary)));
  auto hasBlockDecoration = [](spirv::Opcode opcode,
                               ArrayRef<uint32_t> operands) -> bool {
    if (opcode != spirv::Opcode::OpDecorate || operands.size() != 2)
      return false;
    return operands[1] == static_cast<uint32_t>(spirv::Decoration::Block);
  };
  EXPECT_TRUE(findInstruction(hasBlockDecoration));
}
