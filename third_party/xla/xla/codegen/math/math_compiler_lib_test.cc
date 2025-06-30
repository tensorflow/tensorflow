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

#include "xla/codegen/math/math_compiler_lib.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"

namespace xla::codegen::math {
namespace {

class RemoveFromCompilerUsedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_unique<llvm::LLVMContext>();
    module_ = std::make_unique<llvm::Module>("test_module", *context_);
  }

  llvm::Function* CreateTestFunction(const std::string& name) {
    llvm::FunctionType* func_type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(*context_), {}, false);
    return llvm::Function::Create(func_type, llvm::Function::ExternalLinkage,
                                  name, *module_);
  }

  void CreateCompilerUsedArray(const std::vector<std::string>& function_names) {
    std::vector<llvm::Constant*> elements;

    for (const std::string& name : function_names) {
      llvm::Function* func = CreateTestFunction(name);
      elements.push_back(func);
    }

    llvm::ArrayType* array_type = llvm::ArrayType::get(
        llvm::PointerType::getUnqual(*context_), elements.size());
    llvm::Constant* array_init = llvm::ConstantArray::get(array_type, elements);

    new llvm::GlobalVariable(*module_, array_type, false,
                             llvm::GlobalValue::AppendingLinkage, array_init,
                             "llvm.compiler.used");
  }

  std::vector<std::string> GetCompilerUsedFunctionNames() {
    llvm::GlobalVariable* compiler_used =
        module_->getNamedGlobal("llvm.compiler.used");
    if (!compiler_used) {
      return {};
    }

    llvm::ConstantArray* array =
        llvm::dyn_cast<llvm::ConstantArray>(compiler_used->getInitializer());
    if (!array) {
      return {};
    }

    std::vector<std::string> names;
    for (unsigned i = 0; i < array->getNumOperands(); ++i) {
      llvm::GlobalValue* gv = llvm::dyn_cast<llvm::GlobalValue>(
          array->getOperand(i)->stripPointerCasts());
      if (gv) {
        names.push_back(gv->getName().str());
      }
    }
    return names;
  }

  std::unique_ptr<llvm::LLVMContext> context_;
  std::unique_ptr<llvm::Module> module_;
};

TEST_F(RemoveFromCompilerUsedTest, RemovesSpecifiedFunctions) {
  CreateCompilerUsedArray({"func1", "func2", "func3", "func4"});
  absl::flat_hash_set<absl::string_view> to_remove = {"func2", "func4"};

  RemoveFromCompilerUsed(*module_, to_remove);

  std::vector<std::string> remaining = GetCompilerUsedFunctionNames();
  EXPECT_EQ(remaining.size(), 2) << absl::StrJoin(remaining, ", ");
  EXPECT_TRUE(std::find(remaining.begin(), remaining.end(), "func1") !=
              remaining.end());
  EXPECT_TRUE(std::find(remaining.begin(), remaining.end(), "func3") !=
              remaining.end());
}

TEST_F(RemoveFromCompilerUsedTest, RemovesEntireArrayWhenAllFunctionsRemoved) {
  CreateCompilerUsedArray({"func1", "func2"});
  absl::flat_hash_set<absl::string_view> to_remove = {"func1", "func2"};

  RemoveFromCompilerUsed(*module_, to_remove);

  EXPECT_EQ(module_->getNamedGlobal("llvm.compiler.used"), nullptr);
}

TEST_F(RemoveFromCompilerUsedTest, HandlesNoCompilerUsedArray) {
  // Arrange - no @llvm.compiler.used exists
  absl::flat_hash_set<absl::string_view> to_remove = {"func1"};

  // Act - should not crash
  RemoveFromCompilerUsed(*module_, to_remove);

  EXPECT_EQ(module_->getNamedGlobal("llvm.compiler.used"), nullptr);
}

TEST_F(RemoveFromCompilerUsedTest, DoesNothingWhenNoMatches) {
  CreateCompilerUsedArray({"func1", "func2"});
  absl::flat_hash_set<absl::string_view> to_remove = {"nonexistent"};

  RemoveFromCompilerUsed(*module_, to_remove);

  std::vector<std::string> remaining = GetCompilerUsedFunctionNames();
  EXPECT_EQ(remaining.size(), 2);
  EXPECT_TRUE(std::find(remaining.begin(), remaining.end(), "func1") !=
              remaining.end());
  EXPECT_TRUE(std::find(remaining.begin(), remaining.end(), "func2") !=
              remaining.end());
}

TEST_F(RemoveFromCompilerUsedTest, HandlesEmptyRemovalSet) {
  CreateCompilerUsedArray({"func1", "func2"});
  absl::flat_hash_set<absl::string_view> to_remove = {};

  RemoveFromCompilerUsed(*module_, to_remove);

  std::vector<std::string> remaining = GetCompilerUsedFunctionNames();
  EXPECT_EQ(remaining.size(), 2);
  EXPECT_TRUE(std::find(remaining.begin(), remaining.end(), "func1") !=
              remaining.end());
  EXPECT_TRUE(std::find(remaining.begin(), remaining.end(), "func2") !=
              remaining.end());
}

TEST(MathCompilerLibTest, InlineAndOptPasses) {
  llvm::LLVMContext context;
  llvm::Module module("test", context);
  llvm::Function* f = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getInt32Ty(context), {}, false),
      llvm::GlobalValue::InternalLinkage, "f", &module);
  f->setCallingConv(llvm::CallingConv::Fast);
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(context, "entry", f);
  llvm::ReturnInst::Create(
      context, llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 42),
      entry);

  RunInlineAndOptPasses(module);

  EXPECT_TRUE(module.getNamedGlobal("f") == nullptr);
}

}  // namespace
}  // namespace xla::codegen::math
