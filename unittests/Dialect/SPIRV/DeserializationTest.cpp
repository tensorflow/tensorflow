//===- DeserializationTest.cpp - SPIR-V Deserialization Tests -------------===//
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
// The purpose of this file is to provide negative deserialization tests.
// For positive deserialization tests, please use serialization and
// deserialization for roundtripping.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/SPIRVBinaryUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "gmock/gmock.h"

#include <memory>

using namespace mlir;

using ::testing::StrEq;

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

/// A deserialization test fixture providing minimal SPIR-V building and
/// diagnostic checking utilities.
class DeserializationTest : public ::testing::Test {
protected:
  DeserializationTest() {
    // Register a diagnostic handler to capture the diagnostic so that we can
    // check it later.
    context.getDiagEngine().setHandler([&](Diagnostic diag) {
      diagnostic.reset(new Diagnostic(std::move(diag)));
    });
  }

  /// Performs deserialization and returns the constructed spv.module op.
  Optional<spirv::ModuleOp> deserialize() {
    return spirv::deserialize(binary, &context);
  }

  /// Checks there is a diagnostic generated with the given `errorMessage`.
  void expectDiagnostic(StringRef errorMessage) {
    ASSERT_NE(nullptr, diagnostic.get());

    // TODO(antiagainst): check error location too.
    EXPECT_THAT(diagnostic->str(), StrEq(errorMessage));
  }

  //===--------------------------------------------------------------------===//
  // SPIR-V builder methods
  //===--------------------------------------------------------------------===//

  /// Adds the SPIR-V module header to `binary`.
  void addHeader() { spirv::appendModuleHeader(binary, /*idBound=*/0); }

  /// Adds the SPIR-V instruction into `binary`.
  void addInstruction(spirv::Opcode op, ArrayRef<uint32_t> operands) {
    uint32_t wordCount = 1 + operands.size();
    assert(((wordCount >> 16) == 0) && "word count out of range!");

    uint32_t prefixedOpcode = (wordCount << 16) | static_cast<uint32_t>(op);
    binary.push_back(prefixedOpcode);
    binary.append(operands.begin(), operands.end());
  }

  uint32_t addVoidType() {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpTypeVoid, {id});
    return id;
  }

  uint32_t addIntType(uint32_t bitwidth) {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpTypeInt, {id, bitwidth, /*signedness=*/1});
    return id;
  }

  uint32_t addFunctionType(uint32_t retType, ArrayRef<uint32_t> paramTypes) {
    auto id = nextID++;
    SmallVector<uint32_t, 4> operands;
    operands.push_back(id);
    operands.push_back(retType);
    operands.append(paramTypes.begin(), paramTypes.end());
    addInstruction(spirv::Opcode::OpTypeFunction, operands);
    return id;
  }

  uint32_t addFunction(uint32_t retType, uint32_t fnType) {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpFunction,
                   {retType, id,
                    static_cast<uint32_t>(spirv::FunctionControl::None),
                    fnType});
    return id;
  }

  uint32_t addFunctionEnd() {
    auto id = nextID++;
    addInstruction(spirv::Opcode::OpFunctionEnd, {id});
    return id;
  }

protected:
  SmallVector<uint32_t, 5> binary;
  uint32_t nextID = 1;
  MLIRContext context;
  std::unique_ptr<Diagnostic> diagnostic;
};

//===----------------------------------------------------------------------===//
// Basics
//===----------------------------------------------------------------------===//

TEST_F(DeserializationTest, EmptyModuleFailure) {
  ASSERT_EQ(llvm::None, deserialize());
  expectDiagnostic("SPIR-V binary module must have a 5-word header");
}

TEST_F(DeserializationTest, WrongMagicNumberFailure) {
  addHeader();
  binary.front() = 0xdeadbeef; // Change to a wrong magic number
  ASSERT_EQ(llvm::None, deserialize());
  expectDiagnostic("incorrect magic number");
}

TEST_F(DeserializationTest, OnlyHeaderSuccess) {
  addHeader();
  EXPECT_NE(llvm::None, deserialize());
}

TEST_F(DeserializationTest, ZeroWordCountFailure) {
  addHeader();
  binary.push_back(0); // OpNop with zero word count

  ASSERT_EQ(llvm::None, deserialize());
  expectDiagnostic("word count cannot be zero");
}

TEST_F(DeserializationTest, InsufficientWordFailure) {
  addHeader();
  binary.push_back((2u << 16) |
                   static_cast<uint32_t>(spirv::Opcode::OpTypeVoid));
  // Missing word for type <id>

  ASSERT_EQ(llvm::None, deserialize());
  expectDiagnostic("insufficient words for the last instruction");
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

TEST_F(DeserializationTest, IntTypeMissingSignednessFailure) {
  addHeader();
  addInstruction(spirv::Opcode::OpTypeInt, {nextID++, 32});

  ASSERT_EQ(llvm::None, deserialize());
  expectDiagnostic("OpTypeInt must have bitwidth and signedness parameters");
}

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

TEST_F(DeserializationTest, FunctionMissingEndFailure) {
  addHeader();
  auto voidType = addVoidType();
  auto fnType = addFunctionType(voidType, {});
  addFunction(voidType, fnType);
  // Missing OpFunctionEnd

  ASSERT_EQ(llvm::None, deserialize());
  expectDiagnostic("expected OpFunctionEnd instruction");
}

TEST_F(DeserializationTest, FunctionMissingParameterFailure) {
  addHeader();
  auto voidType = addVoidType();
  auto i32Type = addIntType(32);
  auto fnType = addFunctionType(voidType, {i32Type});
  addFunction(voidType, fnType);
  // Missing OpFunctionParameter

  ASSERT_EQ(llvm::None, deserialize());
  expectDiagnostic("expected OpFunctionParameter instruction");
}
