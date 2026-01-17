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

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "xla/service/gpu/llvm_gpu_backend/amdgpu_backend.h"
#include "xla/service/gpu/llvm_gpu_backend/load_ir_module.h"
#include "xla/tsl/platform/rocm_rocdl_path.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"

namespace xla::gpu {
namespace {

std::string TestIRFile() {
  return tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service", "gpu",
                           "llvm_gpu_backend", "tests_data", "amdgpu.ll");
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
  DebugOptions debug_options;
  debug_options.set_xla_gpu_use_embeded_device_lib(true);
  auto module = LoadIRModule(TestIRFile(), &llvm_context);
  ASSERT_TRUE(HasUndefinedFunctions(*module));
  auto status = amdgpu::LinkROCDLIfNecessary(module.get(), "gfx1200",
                                             debug_options, "<empty>");
  ASSERT_TRUE(status.ok());
  ASSERT_FALSE(HasUndefinedFunctions(*module));
}

TEST(BitcodeLinkTest, TestLinkFromInstallation) {
  llvm::LLVMContext llvm_context;
  DebugOptions debug_options;
  debug_options.set_xla_gpu_use_embeded_device_lib(false);
  auto module = LoadIRModule(TestIRFile(), &llvm_context);
  ASSERT_TRUE(HasUndefinedFunctions(*module));
  auto status = amdgpu::LinkROCDLIfNecessary(module.get(), "gfx1200",
                                             debug_options, tsl::RocdlRoot());
  ASSERT_TRUE(status.ok());
  ASSERT_FALSE(HasUndefinedFunctions(*module));
}

}  // namespace
}  // namespace xla::gpu
