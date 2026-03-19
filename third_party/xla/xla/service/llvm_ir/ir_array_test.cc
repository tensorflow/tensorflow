/* Copyright 2020 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/llvm_ir/ir_array.h"

#include <string>

#include <gtest/gtest.h>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace llvm_ir {
namespace {

class IrArrayTest : public ::testing::Test {
 public:
  IrArrayTest()
      : context_{},
        module_{"IrArrayTest module", context_},
        builder_{context_} {}

  llvm::Function* EmitFunctionAndSetInsertPoint(
      llvm::ArrayRef<llvm::Type*> params) {
    llvm::FunctionType* function_type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(context_), params,
                                /*isVarArg=*/false);
    llvm::Function* function = llvm::Function::Create(
        function_type, llvm::Function::LinkageTypes::ExternalLinkage,
        "test_function", module_);
    llvm::BasicBlock* bb = llvm::BasicBlock::Create(context_, "bb", function);
    builder_.SetInsertPoint(bb);
    return function;
  }

 protected:
  llvm::LLVMContext context_;
  llvm::Module module_;
  llvm::IRBuilder<> builder_;
};

TEST_F(IrArrayTest, TestShapeIsCompatible) {
  xla::Shape a =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 10, 20}, {2, 1, 0});
  xla::Shape b =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 10, 20}, {2, 0, 1});
  xla::Shape c =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 1, 20}, {2, 1, 0});

  xla::Shape d =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 10, 30}, {2, 1, 0});
  xla::Shape e =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 10, 30}, {2, 0, 1});
  xla::Shape f =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {10, 1, 30}, {2, 1, 0});

  EXPECT_TRUE(IrArray::Index::ShapeIsCompatible(a, b));
  EXPECT_TRUE(IrArray::Index::ShapeIsCompatible(a, c));
  EXPECT_FALSE(IrArray::Index::ShapeIsCompatible(a, d));
  EXPECT_FALSE(IrArray::Index::ShapeIsCompatible(a, e));
  EXPECT_FALSE(IrArray::Index::ShapeIsCompatible(a, f));
}

TEST_F(IrArrayTest, EmitArrayElementAddress) {
  llvm::Function* function = EmitFunctionAndSetInsertPoint(
      {builder_.getPtrTy(), builder_.getInt32Ty()});
  llvm::Argument* array_ptr = function->getArg(0);
  llvm::Argument* array_index = function->getArg(1);

  Shape shape = ShapeUtil::MakeShape(F32, {3, 5});
  llvm::Type* type = llvm_ir::ShapeToIrType(shape, module_.getContext());
  IrArray ir_array(array_ptr, type, shape);

  IrArray::Index index(array_index, shape, &builder_);
  ir_array.EmitArrayElementAddress(index, &builder_);
  std::string ir_str = DumpToString(&module_);

  const char* filecheck_pattern = R"(
    CHECK: define void @test_function(ptr %[[ptr:[0-9]+]], i32 %[[idx:[0-9]+]]) {
    CHECK: getelementptr inbounds float, ptr %[[ptr]], i32 %[[idx]]
  )";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_match,
                          RunFileCheck(ir_str, filecheck_pattern));
  EXPECT_TRUE(filecheck_match);
}

TEST_F(IrArrayTest, EmitArrayElementAddressNonLinear) {
  llvm::Function* function = EmitFunctionAndSetInsertPoint(
      {builder_.getPtrTy(), builder_.getInt32Ty()});
  llvm::Argument* array_ptr = function->getArg(0);
  llvm::Argument* array_index = function->getArg(1);

  Shape shape = ShapeUtil::MakeShape(F32, {3, 5});
  llvm::Type* type = llvm_ir::ShapeToIrType(shape, module_.getContext());
  IrArray ir_array(array_ptr, type, shape);

  IrArray::Index index(array_index, shape, &builder_);
  ir_array.EmitArrayElementAddress(index, &builder_, /*name=*/"",
                                   /*use_linear_index=*/false);
  std::string ir_str = DumpToString(&module_);

  const char* filecheck_pattern = R"(
    CHECK: define void @test_function(ptr %[[ptr:[0-9]+]], i32 %[[idx:[0-9]+]]) {
    CHECK: %[[udiv1:[0-9]+]] = udiv i32 %[[idx]], 1
    CHECK: %[[urem:[0-9]+]] = urem i32 %[[udiv1]], 5
    CHECK: %[[udiv2:[0-9]+]] = udiv i32 %[[idx]], 5
    CHECK: getelementptr inbounds [3 x [5 x float]], ptr %0, i32 0, i32 %[[udiv2]], i32 %[[urem]]
  )";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_match,
                          RunFileCheck(ir_str, filecheck_pattern));
  EXPECT_TRUE(filecheck_match);
}

TEST_F(IrArrayTest, EmitArrayElementAddressInt4) {
  llvm::Function* function = EmitFunctionAndSetInsertPoint(
      {builder_.getPtrTy(), builder_.getInt32Ty()});
  llvm::Argument* array_ptr = function->getArg(0);
  llvm::Argument* array_index = function->getArg(1);

  Shape shape = ShapeUtil::MakeShape(S4, {3, 5});
  llvm::Type* type = llvm_ir::ShapeToIrType(shape, module_.getContext());
  IrArray ir_array(array_ptr, type, shape);

  IrArray::Index index(array_index, shape, &builder_);
  llvm::Value* bit_offset;
  ir_array.EmitArrayElementAddress(index, &builder_, /*name=*/"",
                                   /*use_linear_index=*/true,
                                   /*bit_offset=*/&bit_offset);
  std::string ir_str = DumpToString(&module_);

  // The index is divided by 2 and used as an index to the i8 array. A remainder
  // is also computed to calculate bit_offset.
  const char* filecheck_pattern = R"(
    CHECK: define void @test_function(ptr %[[ptr:[0-9]+]], i32 %[[idx:[0-9]+]]) {
    CHECK: %[[rem:[0-9]+]] = urem i32 %[[idx]], 2
    CHECK: %[[div:[0-9]+]] = udiv i32 %[[idx]], 2
    CHECK: getelementptr inbounds i8, ptr %[[ptr]], i32 %[[div]]
  )";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_match,
                          RunFileCheck(ir_str, filecheck_pattern));
  EXPECT_TRUE(filecheck_match);
}

TEST_F(IrArrayTest, EmitArrayElementAddressInt4NonLinear) {
  llvm::Function* function = EmitFunctionAndSetInsertPoint(
      {llvm::PointerType::get(context_, 0), llvm::Type::getInt32Ty(context_),
       llvm::Type::getInt32Ty(context_)});
  llvm::Argument* array_ptr = function->getArg(0);
  llvm::Argument* array_index0 = function->getArg(1);
  llvm::Argument* array_index1 = function->getArg(2);

  Shape shape = ShapeUtil::MakeShape(S4, {3, 5});
  llvm::Type* type = llvm_ir::ShapeToIrType(shape, module_.getContext());
  IrArray ir_array(array_ptr, type, shape);

  IrArray::Index index({array_index0, array_index1}, shape,
                       builder_.getInt32Ty());
  llvm::Value* bit_offset;
  ir_array.EmitArrayElementAddress(index, &builder_, /*name=*/"",
                                   /*use_linear_index=*/false,
                                   /*bit_offset=*/&bit_offset);
  std::string ir_str = DumpToString(&module_);

  // The index is linearized despite use_linear_index=false being passed because
  // non-linear indices are not supported with int4
  const char* filecheck_pattern = R"(
    CHECK: define void @test_function(ptr %[[ptr:[0-9]+]], i32 %[[idx0:[0-9]+]], i32 %[[idx1:[0-9]+]]) {
    CHECK: %[[mul1:[0-9]+]] = mul nuw nsw i32 %[[idx1]], 1
    CHECK: %[[add1:[0-9]+]] = add nuw nsw i32 0, %[[mul1]]
    CHECK: %[[mul2:[0-9]+]] = mul nuw nsw i32 %[[idx0]], 5
    CHECK: %[[add2:[0-9]+]] = add nuw nsw i32 %[[add1]], %[[mul2]]
    CHECK: %[[udiv:[0-9]+]] = udiv i32 %[[add2]], 2
    CHECK: %[[gep:[0-9]+]] = getelementptr inbounds i8, ptr %[[ptr]], i32 %[[udiv]]
  )";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_match,
                          RunFileCheck(ir_str, filecheck_pattern));
  EXPECT_TRUE(filecheck_match);
}

TEST_F(IrArrayTest, EmitReadArrayElementInt4) {
  llvm::Function* function = EmitFunctionAndSetInsertPoint(
      {builder_.getPtrTy(), builder_.getInt32Ty()});
  llvm::Argument* array_ptr = function->getArg(0);
  llvm::Argument* array_index = function->getArg(1);

  Shape shape = ShapeUtil::MakeShape(S4, {3, 5});
  llvm::Type* type = llvm_ir::ShapeToIrType(shape, module_.getContext());
  IrArray ir_array(array_ptr, type, shape);

  IrArray::Index index(array_index, shape, &builder_);
  ir_array.EmitReadArrayElement(index, &builder_);
  std::string ir_str = DumpToString(&module_);

  const char* filecheck_pattern = R"(
    CHECK: define void @test_function(ptr %[[ptr:[0-9]+]], i32 %[[idx0:[0-9]+]]) {

    COM: Calculate the address.
    CHECK: %[[urem:[0-9]+]] = urem i32 %[[idx0]], 2
    CHECK: %[[addr:[0-9]+]] = udiv i32 %[[idx0]], 2
    CHECK: %[[mul:[0-9]+]] = mul i32 %[[urem]], 4
    CHECK: %[[trunc:[0-9]+]] = trunc i32 %[[mul]] to i8
    CHECK: %[[gep:[0-9]+]] = getelementptr inbounds i8, ptr %[[ptr]], i32 %[[addr]]

    COM: Load the element, optionally shift, and truncate.
    CHECK: %[[load:[0-9]+]] = load i8, ptr %[[gep]], align 1
    CHECK: %[[shift:[0-9]+]] = lshr i8 %[[load]], %[[trunc]]
    CHECK: trunc i8 %[[shift]] to i4
  )";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_match,
                          RunFileCheck(ir_str, filecheck_pattern));
  EXPECT_TRUE(filecheck_match);
}

TEST_F(IrArrayTest, EmitWriteArrayElementInt4) {
  llvm::Function* function = EmitFunctionAndSetInsertPoint(
      {builder_.getPtrTy(), builder_.getInt32Ty(), builder_.getIntNTy(4)});
  llvm::Argument* array_ptr = function->getArg(0);
  llvm::Argument* array_index = function->getArg(1);
  llvm::Argument* val_to_write = function->getArg(2);

  Shape shape = ShapeUtil::MakeShape(S4, {3, 5});
  llvm::Type* type = llvm_ir::ShapeToIrType(shape, module_.getContext());
  IrArray ir_array(array_ptr, type, shape);

  IrArray::Index index(array_index, shape, &builder_);
  ir_array.EmitWriteArrayElement(index, val_to_write, &builder_);
  std::string ir_str = DumpToString(&module_);

  const char* filecheck_pattern = R"(
    CHECK: define void @test_function(ptr %[[ptr:[0-9]+]], i32 %[[idx0:[0-9]+]], i4 %[[val:[0-9]+]]) {

    COM: Calculate the address.
    CHECK: %[[urem:[0-9]+]] = urem i32 %[[idx0]], 2
    CHECK: %[[addr:[0-9]+]] = udiv i32 %[[idx0]], 2
    CHECK: %[[mul:[0-9]+]] = mul i32 %[[urem]], 4
    CHECK: %[[trunc:[0-9]+]] = trunc i32 %[[mul]] to i8
    CHECK: %[[gep:[0-9]+]] = getelementptr inbounds i8, ptr %[[ptr]], i32 %[[addr]]

    COM: Load address, replace 4 bits with the value, and write to address.
    CHECK: %[[load:[0-9]+]] = load i8, ptr %[[gep]], align 1
    CHECK: %[[zext:[0-9]+]] = zext i4 %[[val]] to i8
    CHECK: %[[shifted_val:[0-9]+]] = shl i8 %[[zext]], %[[trunc]]
    CHECK: %[[mask:[0-9]+]] = call i8 @llvm.fshl.i8(i8 -16, i8 -16, i8 %[[trunc]])
    CHECK: %[[and:[0-9]+]] = and i8 %[[load]], %[[mask]]
    CHECK: %[[towrite:[0-9]+]] = or i8 %[[and]], %[[shifted_val]]
    CHECK: store i8 %[[towrite]], ptr %[[gep]], align 1
  )";

  TF_ASSERT_OK_AND_ASSIGN(bool filecheck_match,
                          RunFileCheck(ir_str, filecheck_pattern));
  EXPECT_TRUE(filecheck_match);
}

}  // namespace
}  // namespace llvm_ir
}  // namespace xla
