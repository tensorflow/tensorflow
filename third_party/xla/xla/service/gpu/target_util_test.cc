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

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include "tsl/platform/test.h"

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
  module_.setTargetTriple("nvptx");
  EmitCallToTargetIntrinsic(TargetIntrinsicID::kGroupBarrierId,
                            {/*membermask=*/builder_.getInt32(-1)}, {},
                            &builder_);
  builder_.CreateRetVoid();
  EXPECT_FALSE(llvm::verifyModule(module_, &llvm::errs()));
}

TEST_F(TargetUtilTest, AMDGCNGroupBarrier) {
  module_.setTargetTriple("amdgcn");
  EmitCallToTargetIntrinsic(TargetIntrinsicID::kGroupBarrierId, {}, {},
                            &builder_);
  builder_.CreateRetVoid();
  EXPECT_FALSE(llvm::verifyModule(module_, &llvm::errs()));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
