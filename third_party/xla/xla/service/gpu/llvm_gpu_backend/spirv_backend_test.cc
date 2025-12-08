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

#include "xla/service/gpu/llvm_gpu_backend/spirv_backend.h"

#include <gtest/gtest.h>

namespace xla::gpu::spirv {
namespace {

TEST(SpirvBackendTest, TestUnsupportedExtensions) {
  llvm::Triple target_triple("spirv64-unknown-unknown");
  const std::vector<std::string> unsupported_extensions = {
      "SPV_EXT_optnone", "SPV_EXT_arithmetic_fence", "SPV_EXT_hypthetical_name",
      "SPV_KHR_uniform_group_instructions", "SPV_KHR_linkonce_odr"};

  auto extensions =
      RemoveUnsupportedExtensionsFromAll(target_triple, unsupported_extensions);
  auto extensions_set =
      std::set<std::string>(extensions.begin(), extensions.end());

  EXPECT_EQ(extensions_set.find("SPV_EXT_optnone"), extensions_set.end());
  EXPECT_EQ(extensions_set.find("SPV_EXT_arithmetic_fence"),
            extensions_set.end());
  EXPECT_EQ(extensions_set.find("SPV_EXT_hypthetical_name"),
            extensions_set.end());
  EXPECT_EQ(extensions_set.find("SPV_KHR_uniform_group_instructions"),
            extensions_set.end());
  EXPECT_EQ(extensions_set.find("SPV_KHR_linkonce_odr"), extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_KHR_cooperative_matrix"),
            extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_EXT_shader_atomic_float_add"),
            extensions_set.end());
}

}  // namespace
}  // namespace xla::gpu::spirv
