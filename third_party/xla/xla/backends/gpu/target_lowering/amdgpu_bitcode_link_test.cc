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

#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "xla/backends/gpu/target_lowering/amdgpu_backend.h"
#include "xla/backends/gpu/target_lowering/load_ir_module.h"
#include "xla/debug_options_flags.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {

std::string TestIRFile() {
  return tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "backends", "gpu",
                           "target_lowering", "tests_data", "amdgpu.ll");
}

bool HasUndefinedFunctions(const llvm::Module& M) {
  for (const llvm::Function& F : M) {
    if (F.isDeclaration() && !F.isIntrinsic()) {
      return true;
    }
  }
  return false;
}

TEST(BitcodeLinkTest, TestLinkEmbeded) {
  llvm::LLVMContext llvm_context;
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  auto module = LoadIRModule(TestIRFile(), &llvm_context);
  ASSERT_TRUE(HasUndefinedFunctions(*module));
  auto status =
      amdgpu::LinkROCDLIfNecessary(module.get(), "gfx1200", debug_options);
  ASSERT_TRUE(status.ok());
  ASSERT_FALSE(HasUndefinedFunctions(*module));
}

TEST(BitcodeLinkTest, OclcIsaVersion) {
  struct TestCase {
    std::string gfx_version;
    uint64_t isa_version;
  };
  const TestCase kTestCases[] = {
      {"gfx908", 9008},   {"gfx90a", 9010},   {"gfx942", 9402},
      {"gfx950", 9500},   {"gfx1030", 10300}, {"gfx1100", 11000},
      {"gfx1201", 12001}, {"gfx1250", 12500}, {"gfx1250-strict", 12500},
  };

  for (const TestCase& test_case : kTestCases) {
    SCOPED_TRACE(test_case.gfx_version);
    llvm::LLVMContext llvm_context;
    DebugOptions debug_options = GetDebugOptionsFromFlags();
    auto module = LoadIRModule(TestIRFile(), &llvm_context);
    auto status = amdgpu::LinkROCDLIfNecessary(
        module.get(), test_case.gfx_version, debug_options);
    ASSERT_TRUE(status.ok()) << status;

    const llvm::GlobalVariable* isa_version =
        module->getNamedGlobal("__oclc_ISA_version");
    ASSERT_NE(isa_version, nullptr);
    const auto* value =
        llvm::dyn_cast<llvm::ConstantInt>(isa_version->getInitializer());
    ASSERT_NE(value, nullptr);
    EXPECT_EQ(value->getZExtValue(), test_case.isa_version);
  }
}

}  // namespace
}  // namespace xla::gpu
