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

#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/literal.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class PtxKernelE2ETest : public HloPjRtTestBase {};

TEST_F(PtxKernelE2ETest, ScalarAdd) {
  absl::string_view module_str = R"(
    HloModule ptx_test

    ENTRY main {
      a = f32[] constant(3.0)
      b = f32[] constant(4.0)
      ROOT out = f32[] custom-call(a, b),
        custom_call_target="__gpu$xla.gpu.ptx",
        backend_config="{
          name = \"add_kernel\",
          kernel_type = \"ptx\",
          kernel_data = \".version 7.0\\n.target sm_60\\n.address_size 64\\n\\n.visible .entry add_kernel(\\n    .param .u64 input_a,\\n    .param .u64 input_b,\\n    .param .u64 output)\\n{\\n  .reg .f32 a, b, c;\\n  .reg .u64 addr_a, addr_b, addr_out;\\n  \\n  ld.param.u64 addr_a, [input_a];\\n  ld.param.u64 addr_b, [input_b];\\n  ld.param.u64 addr_out, [output];\\n  \\n  ld.global.f32 a, [addr_a];\\n  ld.global.f32 b, [addr_b];\\n  add.f32 c, a, b;\\n  st.global.f32 [addr_out], c;\\n  \\n  ret;\\n}\",
          grid_x = 1, grid_y = 1, grid_z = 1,
          block_x = 1, block_y = 1, block_z = 1,
          shared_mem_bytes = 0,
          output_indices = [2]
        }"
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {}));
  EXPECT_EQ(result.Get<float>({}), 7.0f);
}

TEST_F(PtxKernelE2ETest, TensorAdd) {
  absl::string_view module_str = R"(
    HloModule ptx_tensor_test

    ENTRY main {
      a = f32[4] constant({1.0, 2.0, 3.0, 4.0})
      b = f32[4] constant({5.0, 6.0, 7.0, 8.0})
      ROOT out = f32[4] custom-call(a, b),
        custom_call_target="__gpu$xla.gpu.ptx",
        backend_config="{
          name = \"tensor_add_kernel\",
          kernel_type = \"ptx\",
          kernel_data = \".version 7.0\\n.target sm_60\\n.address_size 64\\n\\n.visible .entry tensor_add_kernel(\\n    .param .u64 input_a,\\n    .param .u64 input_b,\\n    .param .u64 output)\\n{\\n  // Get base pointers\\n  .reg .u64 a_base, b_base, out_base;\\n  ld.param.u64 a_base, [input_a];\\n  ld.param.u64 b_base, [input_b];\\n  ld.param.u64 out_base, [output];\\n  \\n  // Thread ID calculation - just use thread ID directly for this simple case\\n  .reg .u32 tid;\\n  mov.u32 tid, %tid.x;\\n  \\n  // Hard-coded array size = 4\\n  .reg .pred p;\\n  setp.ge.u32 p, tid, 4;\\n  @p bra done;\\n  \\n  // Calculate byte offset (4 bytes per float)\\n  .reg .u64 offset;\\n  cvt.u64.u32 offset, tid;  // Convert tid to 64-bit\\n  mul.lo.u64 offset, offset, 4;  // Each float is 4 bytes\\n  \\n  // Calculate element addresses\\n  .reg .u64 a_addr, b_addr, out_addr;\\n  add.u64 a_addr, a_base, offset;\\n  add.u64 b_addr, b_base, offset;\\n  add.u64 out_addr, out_base, offset;\\n  \\n  // Load input values\\n  .reg .f32 a_val, b_val, result;\\n  ld.global.f32 a_val, [a_addr];\\n  ld.global.f32 b_val, [b_addr];\\n  \\n  // Perform addition\\n  add.f32 result, a_val, b_val;\\n  \\n  // Store result\\n  st.global.f32 [out_addr], result;\\n  \\ndone:\\n  ret;\\n}\",
          grid_x = 1, grid_y = 1, grid_z = 1,
          block_x = 4, block_y = 1, block_z = 1,
          shared_mem_bytes = 0,
          output_indices = [2]
        }"
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {}));

  EXPECT_EQ(result.Get<float>({0}), 6.0f);
  EXPECT_EQ(result.Get<float>({1}), 8.0f);
  EXPECT_EQ(result.Get<float>({2}), 10.0f);
  EXPECT_EQ(result.Get<float>({3}), 12.0f);
}

TEST_F(PtxKernelE2ETest, TensorAddWithoutOutputIndices) {
  absl::string_view module_str = R"(
    HloModule ptx_tensor_test

    ENTRY main {
      a = f32[4] constant({1.0, 2.0, 3.0, 4.0})
      b = f32[4] constant({5.0, 6.0, 7.0, 8.0})
      ROOT out = f32[4] custom-call(a, b),
        custom_call_target="__gpu$xla.gpu.ptx",
        backend_config="{
          name = \"tensor_add_kernel\",
          kernel_type = \"ptx\",
          kernel_data = \".version 7.0\\n.target sm_60\\n.address_size 64\\n\\n.visible .entry tensor_add_kernel(\\n    .param .u64 input_a,\\n    .param .u64 input_b,\\n    .param .u64 output)\\n{\\n  // Get base pointers\\n  .reg .u64 a_base, b_base, out_base;\\n  ld.param.u64 a_base, [input_a];\\n  ld.param.u64 b_base, [input_b];\\n  ld.param.u64 out_base, [output];\\n  \\n  // Thread ID calculation - just use thread ID directly for this simple case\\n  .reg .u32 tid;\\n  mov.u32 tid, %tid.x;\\n  \\n  // Hard-coded array size = 4\\n  .reg .pred p;\\n  setp.ge.u32 p, tid, 4;\\n  @p bra done;\\n  \\n  // Calculate byte offset (4 bytes per float)\\n  .reg .u64 offset;\\n  cvt.u64.u32 offset, tid;  // Convert tid to 64-bit\\n  mul.lo.u64 offset, offset, 4;  // Each float is 4 bytes\\n  \\n  // Calculate element addresses\\n  .reg .u64 a_addr, b_addr, out_addr;\\n  add.u64 a_addr, a_base, offset;\\n  add.u64 b_addr, b_base, offset;\\n  add.u64 out_addr, out_base, offset;\\n  \\n  // Load input values\\n  .reg .f32 a_val, b_val, result;\\n  ld.global.f32 a_val, [a_addr];\\n  ld.global.f32 b_val, [b_addr];\\n  \\n  // Perform addition\\n  add.f32 result, a_val, b_val;\\n  \\n  // Store result\\n  st.global.f32 [out_addr], result;\\n  \\ndone:\\n  ret;\\n}\",
          grid_x = 1, grid_y = 1, grid_z = 1,
          block_x = 4, block_y = 1, block_z = 1,
          shared_mem_bytes = 0
        }"
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {}));

  EXPECT_EQ(result.Get<float>({0}), 6.0f);
  EXPECT_EQ(result.Get<float>({1}), 8.0f);
  EXPECT_EQ(result.Get<float>({2}), 10.0f);
  EXPECT_EQ(result.Get<float>({3}), 12.0f);
}

TEST_F(PtxKernelE2ETest, TensorAddWithNonTrivialOutputIndices) {
  absl::string_view module_str = R"(
    HloModule ptx_tensor_test

    ENTRY main {
      a = f32[4] constant({1.0, 2.0, 3.0, 4.0})
      b = f32[4] constant({5.0, 6.0, 7.0, 8.0})
      ROOT out = f32[4] custom-call(a, b),
        custom_call_target="__gpu$xla.gpu.ptx",
        backend_config="{
          name = \"tensor_add_kernel\",
          kernel_type = \"ptx\",
          kernel_data = \".version 7.0\\n.target sm_60\\n.address_size 64\\n\\n.visible .entry tensor_add_kernel(\\n    .param .u64 input_a,\\n    .param .u64 output,\\n    .param .u64 input_b\\n    )\\n{\\n  // Get base pointers\\n  .reg .u64 a_base, b_base, out_base;\\n  ld.param.u64 a_base, [input_a];\\n  ld.param.u64 out_base, [output];\\n  ld.param.u64 b_base, [input_b];\\n  \\n  // Thread ID calculation - just use thread ID directly for this simple case\\n  .reg .u32 tid;\\n  mov.u32 tid, %tid.x;\\n  \\n  // Hard-coded array size = 4\\n  .reg .pred p;\\n  setp.ge.u32 p, tid, 4;\\n  @p bra done;\\n  \\n  // Calculate byte offset (4 bytes per float)\\n  .reg .u64 offset;\\n  cvt.u64.u32 offset, tid;  // Convert tid to 64-bit\\n  mul.lo.u64 offset, offset, 4;  // Each float is 4 bytes\\n  \\n  // Calculate element addresses\\n  .reg .u64 a_addr, b_addr, out_addr;\\n  add.u64 a_addr, a_base, offset;\\n  add.u64 b_addr, b_base, offset;\\n  add.u64 out_addr, out_base, offset;\\n  \\n  // Load input values\\n  .reg .f32 a_val, b_val, result;\\n  ld.global.f32 a_val, [a_addr];\\n  ld.global.f32 b_val, [b_addr];\\n  \\n  // Perform addition\\n  add.f32 result, a_val, b_val;\\n  \\n  // Store result\\n  st.global.f32 [out_addr], result;\\n  \\ndone:\\n  ret;\\n}\",
          grid_x = 1, grid_y = 1, grid_z = 1,
          block_x = 4, block_y = 1, block_z = 1,
          shared_mem_bytes = 0,
          output_indices = [1]
        }"
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(module_str));
  TF_ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module), {}));

  EXPECT_EQ(result.Get<float>({0}), 6.0f);
  EXPECT_EQ(result.Get<float>({1}), 8.0f);
  EXPECT_EQ(result.Get<float>({2}), 10.0f);
  EXPECT_EQ(result.Get<float>({3}), 12.0f);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
