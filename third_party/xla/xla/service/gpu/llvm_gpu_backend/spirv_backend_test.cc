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

TEST(SpirvBackendTest, TestSPIRVExtensions) {
  auto extensions = SPIRVExtensionsEnumToString(common_spirv_extensions);
  auto extensions_set =
      std::set<std::string>(extensions.begin(), extensions.end());

  EXPECT_NE(extensions_set.find("SPV_EXT_optnone"), extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_KHR_uniform_group_instructions"),
            extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_KHR_linkonce_odr"), extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_KHR_cooperative_matrix"),
            extensions_set.end());
  EXPECT_NE(extensions_set.find("SPV_EXT_shader_atomic_float_add"),
            extensions_set.end());
  EXPECT_EQ(extensions_set.find("SPV_NV_cooperative_matrix"),
            extensions_set.end());
}

}  // namespace
}  // namespace xla::gpu::spirv
