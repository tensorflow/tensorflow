/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/triton/tma_utils.h"

#include <cstdint>
#include <optional>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace xla::gpu {
namespace {

using ::absl::StatusCode;
using ::llvm::SmallVector;
using ::mlir::RankedTensorType;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::triton::FuncOp;
using ::mlir::triton::PointerType;
using ::stream_executor::gpu::TmaDescriptor;
using ::stream_executor::gpu::TmaMetadata;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

TEST(Create2DTmaDescriptorTest, ValidInputReturnCorrectDescriptor) {
  mlir::MLIRContext mlir_context;
  mlir::Builder b(&mlir_context);
  Shape global_shape = ShapeUtil::MakeShape(F32, {256, 128});
  llvm::SmallVector<int64_t, 2> block_shape = {64, 32};
  mlir::Type element_type = b.getF32Type();
  TF_ASSERT_OK_AND_ASSIGN(
      TmaDescriptor tma_desc,
      Create2DTmaDescriptor(global_shape, block_shape, element_type));
  EXPECT_EQ(tma_desc.element_size(), 4);
  EXPECT_EQ(tma_desc.num_dimensions(), 2);
  EXPECT_THAT(tma_desc.global_dims(), ElementsAre(128, 256));
  EXPECT_THAT(tma_desc.global_strides(), ElementsAre(128 * 4));
  EXPECT_THAT(tma_desc.box_dims(), ElementsAre(32, 64));
  EXPECT_THAT(tma_desc.element_strides(), ElementsAre(1, 1));
  EXPECT_EQ(tma_desc.interleave(), TmaDescriptor::TmaInterleave::kNone);
  EXPECT_EQ(tma_desc.swizzle(), TmaDescriptor::TmaSwizzle::k128B);
  EXPECT_EQ(tma_desc.l2_promotion(), TmaDescriptor::TmaL2Promotion::k128B);
  EXPECT_EQ(tma_desc.float_oob_fill(), TmaDescriptor::TmaFloatOobFill::kNone);
}

TEST(Create2DTmaDescriptorTest, BadGlobalShapeFailsGracefully) {
  mlir::MLIRContext mlir_context;
  mlir::Builder b(&mlir_context);
  Shape global_shape = ShapeUtil::MakeShape(F32, {128});
  llvm::SmallVector<int64_t, 2> block_shape = {128, 128};
  mlir::Type element_type = b.getF32Type();
  EXPECT_THAT(Create2DTmaDescriptor(global_shape, block_shape, element_type),
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("expected 2D global shape")));
}

TEST(Create2DTmaDescriptorTest, BadBlockShapeFailsGracefully) {
  mlir::MLIRContext mlir_context;
  mlir::Builder b(&mlir_context);
  Shape global_shape = ShapeUtil::MakeShape(F32, {128, 128});
  llvm::SmallVector<int64_t, 2> block_shape = {128};
  mlir::Type element_type = b.getF32Type();
  EXPECT_THAT(Create2DTmaDescriptor(global_shape, block_shape, element_type),
              StatusIs(StatusCode::kInvalidArgument,
                       HasSubstr("expected 2D block shape")));
}

TEST(Create2DTmaDescriptorTest, SmallBlockShapeFailsGracefully) {
  mlir::MLIRContext mlir_context;
  mlir::Builder b(&mlir_context);
  Shape global_shape = ShapeUtil::MakeShape(F32, {128, 128});
  llvm::SmallVector<int64_t, 2> block_shape = {128, 2};
  mlir::Type element_type = b.getF32Type();
  EXPECT_THAT(Create2DTmaDescriptor(global_shape, block_shape, element_type),
              StatusIs(StatusCode::kFailedPrecondition,
                       HasSubstr("dimension size too small")));
}

class TmaUtilsFixture : public testing::Test {
 public:
  void SetUp() override {
    mlir_context_.loadDialect<mlir::triton::TritonDialect>();
    std::string fn_name = "test_fn";
    auto loc = mlir::NameLoc::get(b_.getStringAttr(fn_name));
    triton_module_ = llvm_ir::CreateMlirModuleOp(loc);
    b_.setInsertionPointToEnd(triton_module_->getBody());
  }

  EmitterLocOpBuilder GetEmitterLocOpBuilder() {
    return EmitterLocOpBuilder(mlir::NameLoc::get(b_.getStringAttr("test_fn")),
                               b_);
  }

  FuncOp CreateTestFunction(EmitterLocOpBuilder& b) {
    std::string fn_name = "test_fn";
    SmallVector<Type, 3> fn_arg_types{
        PointerType::get(b.getF32Type(), mlir::NVVM::kGlobalMemorySpace),
        PointerType::get(b.getF32Type(), mlir::NVVM::kGlobalMemorySpace),
        PointerType::get(b.getF32Type(), mlir::NVVM::kGlobalMemorySpace)};
    auto func_type = b.getFunctionType(fn_arg_types, std::nullopt);
    FuncOp fn = b.create<FuncOp>(fn_name, func_type);
    b.setInsertionPointToStart(fn.addEntryBlock());
    return fn;
  }

 protected:
  mlir::MLIRContext mlir_context_;
  mlir::OpBuilder b_{&mlir_context_};
  mlir::OwningOpRef<mlir::ModuleOp> triton_module_;
};

TEST_F(TmaUtilsFixture,
       EmitTmaDescriptor_ValidInputReturnsCorrectTmaDescriptor) {
  EmitterLocOpBuilder b = GetEmitterLocOpBuilder();
  FuncOp fn = CreateTestFunction(b);
  Value arg = fn.getArgument(0);
  RankedTensorType tensor_type =
      RankedTensorType::get({128, 128}, b.getF32Type());
  Value tma_desc = EmitTmaDescriptor(b, arg, tensor_type);
  EXPECT_EQ(tma_desc.getType(),
            mlir::triton::TensorDescType::get(b.getContext(), tensor_type));
}

TEST_F(TmaUtilsFixture,
       RewriteFunctionForTma_TmaDescriptorsSetCorrectTmaAttribute) {
  EmitterLocOpBuilder b = GetEmitterLocOpBuilder();
  FuncOp fn = CreateTestFunction(b);
  TmaMetadata tma_metadata;
  TF_ASSERT_OK_AND_ASSIGN(
      auto tma_desc,
      TmaDescriptor::Create({128, 128}, {128}, {64, 64}, {1, 1}, 4));
  tma_metadata.arg_index_to_tma_info.insert({0, tma_desc});
  TF_ASSERT_OK_AND_ASSIGN(
      tma_desc, TmaDescriptor::Create({128, 128}, {128}, {64, 64}, {1, 1}, 4));
  tma_metadata.arg_index_to_tma_info.insert({2, tma_desc});

  RewriteFunctionForTma(b, fn, tma_metadata);
  EXPECT_EQ(fn.getArgAttr(0, "tt.nv_tma_desc"), b_.getI32IntegerAttr(1));
  EXPECT_FALSE(fn.getArgAttr(1, "tt.nv_tma_desc"));
  EXPECT_EQ(fn.getArgAttr(2, "tt.nv_tma_desc"), b_.getI32IntegerAttr(1));
}

TEST_F(TmaUtilsFixture,
       RewriteFunctionForTma_NoTmaMetadataDoesNotSetTmaAttribute) {
  EmitterLocOpBuilder b = GetEmitterLocOpBuilder();
  FuncOp fn = CreateTestFunction(b);
  RewriteFunctionForTma(b, fn, std::nullopt);
  EXPECT_FALSE(fn.getArgAttr(0, "tt.nv_tma_desc"));
  EXPECT_FALSE(fn.getArgAttr(1, "tt.nv_tma_desc"));
  EXPECT_FALSE(fn.getArgAttr(2, "tt.nv_tma_desc"));
}

TEST_F(TmaUtilsFixture,
       RewriteFunctionForTma_EmptyTmaMetadataDoesNotSetTmaAttribute) {
  EmitterLocOpBuilder b = GetEmitterLocOpBuilder();
  FuncOp fn = CreateTestFunction(b);
  TmaMetadata tma_metadata;
  RewriteFunctionForTma(b, fn, tma_metadata);
  EXPECT_FALSE(fn.getArgAttr(0, "tt.nv_tma_desc"));
  EXPECT_FALSE(fn.getArgAttr(1, "tt.nv_tma_desc"));
  EXPECT_FALSE(fn.getArgAttr(2, "tt.nv_tma_desc"));
}

}  // namespace
}  // namespace xla::gpu
