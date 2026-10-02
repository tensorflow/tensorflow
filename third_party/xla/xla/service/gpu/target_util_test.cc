/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/target_util.h"

#include <gtest/gtest.h>
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

class TargetUtilTest : public testing::Test {
 public:
  TargetUtilTest() : module_("test", ctx_), builder_(ctx_) {}

 protected:
  void SetUp() override {
    auto fn = llvm::Function::Create(
        llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_), {}),
        llvm::Function::LinkageTypes::ExternalLinkage, "fn", module_);
    auto block = llvm::BasicBlock::Create(ctx_, "blk", fn);
    builder_.SetInsertPoint(block);
  }

  llvm::LLVMContext ctx_;
  llvm::Module module_;
  llvm::IRBuilder<> builder_;
};

TEST_F(TargetUtilTest, NVPTXGroupBarrier) {
  module_.setTargetTriple(llvm::Triple("nvptx"));
  EmitCallToTargetIntrinsic(TargetIntrinsicID::kGroupBarrierId,
                            {/*membermask=*/builder_.getInt32(-1)}, {},
                            &builder_);
  builder_.CreateRetVoid();
  EXPECT_FALSE(llvm::verifyModule(module_, &llvm::errs()));
}

TEST_F(TargetUtilTest, AMDGCNGroupBarrier) {
  module_.setTargetTriple(llvm::Triple("amdgcn"));
  EmitCallToTargetIntrinsic(TargetIntrinsicID::kGroupBarrierId, {}, {},
                            &builder_);
  builder_.CreateRetVoid();
  EXPECT_FALSE(llvm::verifyModule(module_, &llvm::errs()));
}

TEST(TargetUtil, ObtainDeviceFunctionNameExp) {
  llvm::Triple triple("nvptx64-unknown-unknown");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kExp,
                                     /*output_type=*/F32, triple),
            "__nv_expf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kExp,
                                     /*output_type=*/BF16, triple),
            "__nv_fast_expf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kExp,
                                     /*output_type=*/F16, triple),
            "__nv_fast_expf");
}

TEST(TargetUtil, ObtainDeviceFunctionNameLog) {
  llvm::Triple triple("nvptx64-unknown-unknown");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kLog,
                                     /*output_type=*/F32, triple),
            "__nv_logf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kLog,
                                     /*output_type=*/BF16, triple),
            "__nv_fast_logf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kLog,
                                     /*output_type=*/F16, triple),
            "__nv_fast_logf");
}

TEST(TargetUtil, ObtainDeviceFunctionNameAtan) {
  llvm::Triple nvptx_triple("nvptx64-unknown-unknown");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kAtan,
                                     /*output_type=*/F32, nvptx_triple),
            "__nv_atanf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kAtan,
                                     /*output_type=*/F16, nvptx_triple),
            "__nv_atanf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kAtan,
                                     /*output_type=*/BF16, nvptx_triple),
            "__nv_atanf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kAtan,
                                     /*output_type=*/F64, nvptx_triple),
            "__nv_atan");

  llvm::Triple amdgpu_triple("amdgcn-amd-amdhsa");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kAtan,
                                     /*output_type=*/F32, amdgpu_triple),
            "__ocml_atan_f32");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kAtan,
                                     /*output_type=*/F16, amdgpu_triple),
            "__ocml_atan_f16");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kAtan,
                                     /*output_type=*/BF16, amdgpu_triple),
            "__ocml_atan_f32");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kAtan,
                                     /*output_type=*/F64, amdgpu_triple),
            "__ocml_atan_f64");

  llvm::Triple spir_triple("spir64-unknown-unknown");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kAtan,
                                     /*output_type=*/F32, spir_triple),
            "_Z16__spirv_ocl_atanf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kAtan,
                                     /*output_type=*/F16, spir_triple),
            "_Z16__spirv_ocl_atanf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kAtan,
                                     /*output_type=*/BF16, spir_triple),
            "_Z16__spirv_ocl_atanf");
  EXPECT_EQ(ObtainDeviceFunctionName(TargetDeviceFunctionID::kAtan,
                                     /*output_type=*/F64, spir_triple),
            "_Z16__spirv_ocl_atand");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
