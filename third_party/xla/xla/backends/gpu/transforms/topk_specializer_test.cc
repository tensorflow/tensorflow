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

#include "xla/backends/gpu/transforms/topk_specializer.h"

#include <stddef.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/backends/gpu/runtime/raft_select_k_utils.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/topk_rewriter.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"

namespace xla::gpu {
namespace {

using ::testing::Combine;
using ::testing::Values;

// Params:
//  - n_kb: number of elements in kilobytes.
//  - k: number of elements to return.
//  - batch_size
//  - dtype
using ParameterizedInterface =
    ::testing::WithParamInterface<std::tuple<int, int, int, absl::string_view>>;

class TopkTest : public HloPjRtGpuTestBase, public ParameterizedInterface {
 protected:
  TopkTest()
      : HloPjRtGpuTestBase(
            HloTestBaseOptions{.verifier_layout_sensitive = true}) {}

  absl::StatusOr<std::unique_ptr<HloModule>> TopkHlo(
      int n, int k, int batch_size, absl::string_view dtype) const {
    return ParseAndReturnVerifiedModule(absl::Substitute(
        R"(
      %compare {
        %p.1.lhs.40628 = s32[] parameter(2)
        %p.1.rhs.40629 = s32[] parameter(3)
        %constant.40630 = pred[] constant(true)
        %broadcast.40631 = pred[] broadcast(pred[] %constant.40630), dimensions={}
        %p.0.lhs.40626 = $3[] parameter(0)
        %p.0.rhs.40627 = $3[] parameter(1)
        %compare.40632 = pred[] compare($3[] %p.0.lhs.40626, $3[] %p.0.rhs.40627), direction=GT, order=TOTAL
        ROOT %select.40633 = pred[] select(pred[] %broadcast.40631, pred[] %compare.40632, pred[] %broadcast.40631)
      }

      ENTRY top_k {
        %arg = $3[$2,$0] parameter(0)
        ROOT %result = ($3[$2,$1], s32[$2,$1]) custom-call(%arg), custom_call_target="TopK", to_apply=%compare
      }
    )",
        n, k, batch_size, dtype));
  }
};

TEST_P(TopkTest, ProducesCorrectResult) {
  // TODO(intel-tf): Remove this check once specialization for SYCL/oneAPI
  // backend is added.
  if (device_description().gpu_compute_capability().IsOneAPI()) {
    GTEST_SKIP() << "OneAPI does not support TopK custom call.";
  }
  const auto [n_kb, k, batch_size, dtype] = GetParam();
  const size_t n = n_kb * 1024;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> topk_module,
                       TopkHlo(n, k, batch_size, dtype));
  std::unique_ptr<HloModule> reference_module = topk_module->Clone("reference");
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      TopkSpecializer(device_description()).Run(topk_module.get()));
  ASSERT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(changed, TopkDecomposer().Run(reference_module.get()));
  ASSERT_TRUE(changed);
  // Disable TopkRewriter, otherwise it will rewrite sort+slice back to TopK
  // custom call and make test and reference modules identical.
  reference_module->mutable_config()
      .mutable_debug_options()
      .add_xla_disable_hlo_passes("topk-rewriter");

  EXPECT_TRUE(RunAndCompareTwoModules(
      std::move(topk_module), std::move(reference_module), std::nullopt));
}

INSTANTIATE_TEST_SUITE_P(
    TopkTests, TopkTest,
    Combine(
        /*n_kb=*/Values(1, 8, 12, 32),
        /*k=*/Values(1, 2, 4, 8, 16, 7, 12),
        /*batch_size=*/Values(1, 16, 32, 64, 128),
        /*dtype=*/Values(absl::string_view("f32"), "bf16")),
    [](const auto& info) {
      return absl::Substitute("n$0KiB_k$1_batch_size$2_$3",
                              std::get<0>(info.param), std::get<1>(info.param),
                              std::get<2>(info.param), std::get<3>(info.param));
    });

TEST_F(TopkTest, PreservesBackendConfig) {
  const char* hlo = R"(
    HloModule m

    %compare-gt.1 {
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      ROOT compare = pred[] compare(p.0.lhs, p.0.rhs), direction=GT, order=TOTAL
    }

    ENTRY top_k {
      arg = f32[8,1024] parameter(0)
      ROOT result = (f32[8,64], s32[8,64]) custom-call(arg), custom_call_target="TopK", called_computations={%compare-gt.1}, backend_config={is_stable = false, order = "TOTAL"}
    }
  )";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(bool changed,
                       TopkSpecializer(device_description()).Run(module.get()));
  if (!device_description().gpu_compute_capability().IsCuda()) {
    // RAFT is CUDA-only. Without RAFT, k=64 exceeds the default max_k=16
    // for SmallBufferOptimization, so the pass should gracefully do nothing.
    ASSERT_FALSE(changed);
    return;
  }
  ASSERT_TRUE(changed);

  // Verify the new __gpu$TopK custom call still has the backend config.
  // k=64 would normally fail the SmallBufferOptimization max_k=16 check,
  // but should succeed because is_stable=false triggers the RAFT path.
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* custom_call = root->operand(0)->operand(0);
  EXPECT_EQ(custom_call->custom_call_target(), "__gpu$TopK");
  EXPECT_EQ(custom_call->raw_backend_config_string(),
            "{is_stable = false, order = \"TOTAL\"}");
}

TEST_F(TopkTest, RewriteStableTopKF32ToUint64) {
  const char* hlo = R"(
    HloModule m

    %compare-gt.1 {
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      ROOT compare = pred[] compare(p.0.lhs, p.0.rhs), direction=GT, order=TOTAL
    }

    ENTRY top_k {
      arg = f32[8,1024] parameter(0)
      ROOT result = (f32[8,32], s32[8,32]) custom-call(arg), custom_call_target="TopK", called_computations={%compare-gt.1}, backend_config={is_stable = true, order = "TOTAL"}
    }
  )";

  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options()
      .set_xla_gpu_experimental_enable_raft_for_stable_topk(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo, config));

  if (!device_description().gpu_compute_capability().IsCuda()) {
    GTEST_SKIP() << "RAFT is CUDA-only.";
  }

  ASSERT_OK_AND_ASSIGN(bool changed,
                       TopkSpecializer(device_description()).Run(module.get()));
  ASSERT_TRUE(changed);

  const se::CudaComputeCapability* cuda_cc =
      device_description().gpu_compute_capability().cuda_compute_capability();
  ASSERT_NE(cuda_cc, nullptr);
  const int expected_sm_count =
      GetRaftSelectKMaxSmCount(*cuda_cc, device_description().core_count());
  ASSERT_OK_AND_ASSIGN(const int64_t expected_scratch_bytes,
                       EstimateRaftSelectKScratchBytesUInt64(
                           /*batch=*/8, /*n=*/1024, /*k=*/32, expected_sm_count,
                           RaftScratchPhaseMode::kSumAcrossPhases));

  const std::string check_pattern = absl::Substitute(
      R"(
// CHECK-LABEL: ENTRY %top_k
// CHECK: %[[ARG:[^ ]+]] = f32[8,1024]{{.*}} parameter(0)

// 1. Bitcast to U32 and S32
// CHECK: %[[VAL_U32:[^ ]+]] = u32[8,1024]{{.*}} bitcast-convert(%[[ARG]])
// CHECK: %[[VAL_S32:[^ ]+]] = s32[8,1024]{{.*}} bitcast-convert(%[[ARG]])

// 2. Pure Bitwise Radix Flip (F32 -> U32)
// CHECK: %[[SRA:[^ ]+]] = s32[8,1024]{{.*}} shift-right-arithmetic(%[[VAL_S32]], {{.*}})
// CHECK: %[[SRA_U32:[^ ]+]] = u32[8,1024]{{.*}} bitcast-convert(%[[SRA]])
// CHECK: %[[MASK:[^ ]+]] = u32[8,1024]{{.*}} or(%[[SRA_U32]], {{.*}})
// CHECK: %[[RADIX_KEY:[^ ]+]] = u32[8,1024]{{.*}} xor(%[[VAL_U32]], %[[MASK]])

// 3. Pack into U64
// CHECK: %[[KEY_U64:[^ ]+]] = u64[8,1024]{{.*}} convert(%[[RADIX_KEY]])
// CHECK: %[[SHIFT_LEFT:[^ ]+]] = u64[8,1024]{{.*}} shift-left(%[[KEY_U64]], {{.*}})
// CHECK: %[[IOTA:[^ ]+]] = u32[8,1024]{{.*}} iota(), iota_dimension=1
// CHECK: %[[SUBTRACT:[^ ]+]] = u32[8,1024]{{.*}} subtract({{.*}}, %[[IOTA]])
// CHECK: %[[PACKED:[^ ]+]] = u64[8,1024]{{.*}} or(%[[SHIFT_LEFT]], {{.*}})

// 4. CustomCall (__gpu$$TopK)
// CHECK: %[[CUSTOM_CALL:[^ ]+]] = (u64[8,32]{{.*}}, s32[8,32]{{.*}}, u8[$0]{{.*}}) custom-call(%[[PACKED]]), custom_call_target="__gpu$$TopK", api_version=API_VERSION_TYPED_FFI, backend_config={is_stable = false, order = "TOTAL"}

// 5. Unpack U64 -> U32
// CHECK: %[[SRL:[^ ]+]] = u64[8,32]{{.*}} shift-right-logical(%[[CUSTOM_CALL]]#0, {{.*}})
// CHECK: %[[UNPACK_U32:[^ ]+]] = u32[8,32]{{.*}} convert(%[[SRL]])

// 6. Pure Bitwise Radix Unflip (U32 -> F32)
// CHECK: %[[MSB:[^ ]+]] = u32[8,32]{{.*}} shift-right-logical(%[[UNPACK_U32]], {{.*}})
// CHECK: %[[MSB_MINUS_ONE:[^ ]+]] = u32[8,32]{{.*}} subtract(%[[MSB]], {{.*}})
// CHECK: %[[UNMASK:[^ ]+]] = u32[8,32]{{.*}} or(%[[MSB_MINUS_ONE]], {{.*}})
// CHECK: %[[UNFLIPPED:[^ ]+]] = u32[8,32]{{.*}} xor(%[[UNPACK_U32]], %[[UNMASK]])
// CHECK: %[[F32_OUT:[^ ]+]] = f32[8,32]{{.*}} bitcast-convert(%[[UNFLIPPED]])

// 7. Tuple output
// CHECK: ROOT {{.*}} tuple({{.*}}%[[F32_OUT]], {{.*}}%[[CUSTOM_CALL]]#1)
  )",
      expected_scratch_bytes);

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), check_pattern));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(TopkTest, RewriteStableTopKBF16ToUint64) {
  const char* hlo = R"(
    HloModule m

    %compare-gt.1 {
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      p.0.lhs = bf16[] parameter(0)
      p.0.rhs = bf16[] parameter(1)
      ROOT compare = pred[] compare(p.0.lhs, p.0.rhs), direction=GT, order=TOTAL
    }

    ENTRY top_k {
      arg = bf16[8,65540] parameter(0)
      ROOT result = (bf16[8,32], s32[8,32]) custom-call(arg), custom_call_target="TopK", called_computations={%compare-gt.1}, backend_config={is_stable = true, order = "TOTAL"}
    }
  )";

  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options()
      .set_xla_gpu_experimental_enable_raft_for_stable_topk(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo, config));

  if (!device_description().gpu_compute_capability().IsCuda()) {
    GTEST_SKIP() << "RAFT is CUDA-only.";
  }

  ASSERT_OK_AND_ASSIGN(bool changed,
                       TopkSpecializer(device_description()).Run(module.get()));
  ASSERT_TRUE(changed);

  const se::CudaComputeCapability* cuda_cc =
      device_description().gpu_compute_capability().cuda_compute_capability();
  ASSERT_NE(cuda_cc, nullptr);
  const int expected_sm_count =
      GetRaftSelectKMaxSmCount(*cuda_cc, device_description().core_count());
  ASSERT_OK_AND_ASSIGN(
      const int64_t expected_scratch_bytes,
      EstimateRaftSelectKScratchBytesUInt64(
          /*batch=*/8, /*n=*/65540, /*k=*/32, expected_sm_count,
          RaftScratchPhaseMode::kSumAcrossPhases));

  const std::string check_pattern = absl::Substitute(
      R"(
// CHECK-LABEL: ENTRY %top_k
// CHECK: %[[ARG:[^ ]+]] = bf16[8,65540]{{.*}} parameter(0)

// 1. Convert BF16 -> F32, then Bitcast to U32 and S32
// CHECK: %[[F32_VAL:[^ ]+]] = f32[8,65540]{{.*}} convert(%[[ARG]])
// CHECK: %[[VAL_U32:[^ ]+]] = u32[8,65540]{{.*}} bitcast-convert(%[[F32_VAL]])
// CHECK: %[[VAL_S32:[^ ]+]] = s32[8,65540]{{.*}} bitcast-convert(%[[F32_VAL]])

// 2. Pure Bitwise Radix Flip (F32 -> U32)
// CHECK: %[[SRA:[^ ]+]] = s32[8,65540]{{.*}} shift-right-arithmetic(%[[VAL_S32]], {{.*}})
// CHECK: %[[SRA_U32:[^ ]+]] = u32[8,65540]{{.*}} bitcast-convert(%[[SRA]])
// CHECK: %[[MASK:[^ ]+]] = u32[8,65540]{{.*}} or(%[[SRA_U32]], {{.*}})
// CHECK: %[[RADIX_KEY:[^ ]+]] = u32[8,65540]{{.*}} xor(%[[VAL_U32]], %[[MASK]])

// 3. Pack into U64
// CHECK: %[[KEY_U64:[^ ]+]] = u64[8,65540]{{.*}} convert(%[[RADIX_KEY]])
// CHECK: %[[SHIFT_LEFT:[^ ]+]] = u64[8,65540]{{.*}} shift-left(%[[KEY_U64]], {{.*}})
// CHECK: %[[IOTA:[^ ]+]] = u32[8,65540]{{.*}} iota(), iota_dimension=1
// CHECK: %[[SUBTRACT:[^ ]+]] = u32[8,65540]{{.*}} subtract({{.*}}, %[[IOTA]])
// CHECK: %[[PACKED:[^ ]+]] = u64[8,65540]{{.*}} or(%[[SHIFT_LEFT]], {{.*}})

// 4. CustomCall (__gpu$$TopK)
// CHECK: %[[CUSTOM_CALL:[^ ]+]] = (u64[8,32]{{.*}}, s32[8,32]{{.*}}, u8[$0]{{.*}}) custom-call(%[[PACKED]]), custom_call_target="__gpu$$TopK", api_version=API_VERSION_TYPED_FFI, backend_config={is_stable = false, order = "TOTAL"}

// 5. Unpack U64 -> U32
// CHECK: %[[SRL:[^ ]+]] = u64[8,32]{{.*}} shift-right-logical(%[[CUSTOM_CALL]]#0, {{.*}})
// CHECK: %[[UNPACK_U32:[^ ]+]] = u32[8,32]{{.*}} convert(%[[SRL]])

// 6. Pure Bitwise Radix Unflip (U32 -> F32)
// CHECK: %[[MSB:[^ ]+]] = u32[8,32]{{.*}} shift-right-logical(%[[UNPACK_U32]], {{.*}})
// CHECK: %[[MSB_MINUS_ONE:[^ ]+]] = u32[8,32]{{.*}} subtract(%[[MSB]], {{.*}})
// CHECK: %[[UNMASK:[^ ]+]] = u32[8,32]{{.*}} or(%[[MSB_MINUS_ONE]], {{.*}})
// CHECK: %[[UNFLIPPED:[^ ]+]] = u32[8,32]{{.*}} xor(%[[UNPACK_U32]], %[[UNMASK]])
// CHECK: %[[F32_OUT:[^ ]+]] = f32[8,32]{{.*}} bitcast-convert(%[[UNFLIPPED]])

// 7. Convert F32 -> BF16 and Tuple output
// CHECK: %[[BF16_OUT:[^ ]+]] = bf16[8,32]{{.*}} convert(%[[F32_OUT]])
// CHECK: ROOT {{.*}} tuple({{.*}}%[[BF16_OUT]], {{.*}}%[[CUSTOM_CALL]]#1)
  )",
      expected_scratch_bytes);

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), check_pattern));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(TopkTest, RewriteStableTopKDefaultAndDisableFlag) {
  const char* hlo = R"(
    HloModule m

    %compare-gt.1 {
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      ROOT compare = pred[] compare(p.0.lhs, p.0.rhs), direction=GT, order=TOTAL
    }

    ENTRY top_k {
      arg = f32[8,1024] parameter(0)
      ROOT result = (f32[8,32], s32[8,32]) custom-call(arg), custom_call_target="TopK", called_computations={%compare-gt.1}, backend_config={is_stable = true, order = "TOTAL"}
    }
  )";
  if (!device_description().gpu_compute_capability().IsCuda()) {
    GTEST_SKIP() << "RAFT is CUDA-only.";
  }
  // Enabled by default.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));
  ASSERT_OK_AND_ASSIGN(bool changed,
                       TopkSpecializer(device_description()).Run(module.get()));
  EXPECT_TRUE(changed);
  // Can still be disabled via flag.
  HloModuleConfig disabled_config = GetModuleConfigForTest();
  disabled_config.mutable_debug_options()
      .set_xla_gpu_experimental_enable_raft_for_stable_topk(false);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> disabled_module,
                       ParseAndReturnVerifiedModule(hlo, disabled_config));
  ASSERT_OK_AND_ASSIGN(
      changed,
      TopkSpecializer(device_description()).Run(disabled_module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(TopkTest, RewriteStablePartialOrderTopKF32ToUint64) {
  const char* hlo = R"(
    HloModule m

    %compare-gt.1 {
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      ROOT compare = pred[] compare(p.0.lhs, p.0.rhs), direction=GT
    }

    ENTRY top_k {
      arg = f32[8,1024] parameter(0)
      ROOT result = (f32[8,32], s32[8,32]) custom-call(arg), custom_call_target="TopK", called_computations={%compare-gt.1}, backend_config={is_stable = true, order = "PARTIAL"}
    }
  )";

  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options()
      .set_xla_gpu_experimental_enable_raft_for_stable_topk(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo, config));

  if (!device_description().gpu_compute_capability().IsCuda()) {
    GTEST_SKIP() << "RAFT is CUDA-only.";
  }

  ASSERT_OK_AND_ASSIGN(bool changed,
                       TopkSpecializer(device_description()).Run(module.get()));
  ASSERT_TRUE(changed);

  const se::CudaComputeCapability* cuda_cc =
      device_description().gpu_compute_capability().cuda_compute_capability();
  ASSERT_NE(cuda_cc, nullptr);
  const int expected_sm_count =
      GetRaftSelectKMaxSmCount(*cuda_cc, device_description().core_count());
  ASSERT_OK_AND_ASSIGN(const int64_t expected_scratch_bytes,
                       EstimateRaftSelectKScratchBytesUInt64(
                           /*batch=*/8, /*n=*/1024, /*k=*/32, expected_sm_count,
                           RaftScratchPhaseMode::kSumAcrossPhases));

  const std::string check_pattern = absl::Substitute(
      R"(
// CHECK-LABEL: ENTRY %top_k
// CHECK: %[[ARG:[^ ]+]] = f32[8,1024]{{.*}} parameter(0)

// 1. Bitcast to U32 and S32
// CHECK: %[[VAL_U32:[^ ]+]] = u32[8,1024]{{.*}} bitcast-convert(%[[ARG]])
// CHECK: %[[VAL_S32:[^ ]+]] = s32[8,1024]{{.*}} bitcast-convert(%[[ARG]])

// 2. Pure Bitwise Radix Flip (F32 -> U32)
// CHECK: %[[SRA:[^ ]+]] = s32[8,1024]{{.*}} shift-right-arithmetic(%[[VAL_S32]], {{.*}})
// CHECK: %[[SRA_U32:[^ ]+]] = u32[8,1024]{{.*}} bitcast-convert(%[[SRA]])
// CHECK: %[[MASK:[^ ]+]] = u32[8,1024]{{.*}} or(%[[SRA_U32]], {{.*}})
// CHECK: %[[RADIX_KEY:[^ ]+]] = u32[8,1024]{{.*}} xor(%[[VAL_U32]], %[[MASK]])

// 3. Canonicalize -0.0f and pack into U64 with 31-bit reversed iota and is_neg_zero bit
// CHECK: %[[IS_NEG_ZERO_PRED:[^ ]+]] = pred[8,1024]{{.*}} compare(%[[VAL_U32]], {{.*}}), direction=EQ
// CHECK: %[[IS_NEG_ZERO:[^ ]+]] = u32[8,1024]{{.*}} convert(%[[IS_NEG_ZERO_PRED]])
// CHECK: %[[PARTIAL_KEY:[^ ]+]] = u32[8,1024]{{.*}} add(%[[RADIX_KEY]], %[[IS_NEG_ZERO]])
// CHECK: %[[KEY_U64:[^ ]+]] = u64[8,1024]{{.*}} convert(%[[PARTIAL_KEY]])
// CHECK: %[[SHIFT_LEFT:[^ ]+]] = u64[8,1024]{{.*}} shift-left(%[[KEY_U64]], {{.*}})
// CHECK: %[[IOTA:[^ ]+]] = u32[8,1024]{{.*}} iota(), iota_dimension=1
// CHECK: %[[IOTA_NEG:[^ ]+]] = u32[8,1024]{{.*}} subtract({{.*}}, %[[IOTA]])
// CHECK: %[[IOTA_SHIFTED:[^ ]+]] = u32[8,1024]{{.*}} shift-left(%[[IOTA_NEG]], {{.*}})
// CHECK: %[[LOW_U32:[^ ]+]] = u32[8,1024]{{.*}} or(%[[IOTA_SHIFTED]], %[[IS_NEG_ZERO]])
// CHECK: %[[LOW_U64:[^ ]+]] = u64[8,1024]{{.*}} convert(%[[LOW_U32]])
// CHECK: %[[PACKED:[^ ]+]] = u64[8,1024]{{.*}} or(%[[SHIFT_LEFT]], %[[LOW_U64]])

// 4. CustomCall (__gpu$$TopK)
// CHECK: %[[CUSTOM_CALL:[^ ]+]] = (u64[8,32]{{.*}}, s32[8,32]{{.*}}, u8[$0]{{.*}}) custom-call(%[[PACKED]]), custom_call_target="__gpu$$TopK", api_version=API_VERSION_TYPED_FFI, backend_config={is_stable = false, order = "PARTIAL"}

// 5. Unpack U64 -> U32 and restore -0.0f radix key
// CHECK: %[[SRL:[^ ]+]] = u64[8,32]{{.*}} shift-right-logical(%[[CUSTOM_CALL]]#0, {{.*}})
// CHECK: %[[PARTIAL_UNPACK_U32:[^ ]+]] = u32[8,32]{{.*}} convert(%[[SRL]])
// CHECK: %[[UNPACK_LOW_U32:[^ ]+]] = u32[8,32]{{.*}} convert(%[[CUSTOM_CALL]]#0)
// CHECK: %[[UNPACK_IS_NEG_ZERO:[^ ]+]] = u32[8,32]{{.*}} and(%[[UNPACK_LOW_U32]], {{.*}})
// CHECK: %[[UNPACK_U32:[^ ]+]] = u32[8,32]{{.*}} subtract(%[[PARTIAL_UNPACK_U32]], %[[UNPACK_IS_NEG_ZERO]])

// 6. Pure Bitwise Radix Unflip (U32 -> F32)
// CHECK: %[[MSB:[^ ]+]] = u32[8,32]{{.*}} shift-right-logical(%[[UNPACK_U32]], {{.*}})
// CHECK: %[[MSB_MINUS_ONE:[^ ]+]] = u32[8,32]{{.*}} subtract(%[[MSB]], {{.*}})
// CHECK: %[[UNMASK:[^ ]+]] = u32[8,32]{{.*}} or(%[[MSB_MINUS_ONE]], {{.*}})
// CHECK: %[[UNFLIPPED:[^ ]+]] = u32[8,32]{{.*}} xor(%[[UNPACK_U32]], %[[UNMASK]])
// CHECK: %[[F32_OUT:[^ ]+]] = f32[8,32]{{.*}} bitcast-convert(%[[UNFLIPPED]])

// 7. Tuple output
// CHECK: ROOT {{.*}} tuple({{.*}}%[[F32_OUT]], {{.*}}%[[CUSTOM_CALL]]#1)
  )",
      expected_scratch_bytes);

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), check_pattern));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(TopkTest, RewriteStablePartialOrderTopKBF16ToUint64) {
  const char* hlo = R"(
    HloModule m

    %compare-gt.1 {
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      p.0.lhs = bf16[] parameter(0)
      p.0.rhs = bf16[] parameter(1)
      ROOT compare = pred[] compare(p.0.lhs, p.0.rhs), direction=GT
    }

    ENTRY top_k {
      arg = bf16[8,65540] parameter(0)
      ROOT result = (bf16[8,32], s32[8,32]) custom-call(arg), custom_call_target="TopK", called_computations={%compare-gt.1}, backend_config={is_stable = true, order = "PARTIAL"}
    }
  )";

  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options()
      .set_xla_gpu_experimental_enable_raft_for_stable_topk(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo, config));

  if (!device_description().gpu_compute_capability().IsCuda()) {
    GTEST_SKIP() << "RAFT is CUDA-only.";
  }

  ASSERT_OK_AND_ASSIGN(bool changed,
                       TopkSpecializer(device_description()).Run(module.get()));
  ASSERT_TRUE(changed);

  const se::CudaComputeCapability* cuda_cc =
      device_description().gpu_compute_capability().cuda_compute_capability();
  ASSERT_NE(cuda_cc, nullptr);
  const int expected_sm_count =
      GetRaftSelectKMaxSmCount(*cuda_cc, device_description().core_count());
  ASSERT_OK_AND_ASSIGN(
      const int64_t expected_scratch_bytes,
      EstimateRaftSelectKScratchBytesUInt64(
          /*batch=*/8, /*n=*/65540, /*k=*/32, expected_sm_count,
          RaftScratchPhaseMode::kSumAcrossPhases));

  const std::string check_pattern = absl::Substitute(
      R"(
// CHECK-LABEL: ENTRY %top_k
// CHECK: %[[ARG:[^ ]+]] = bf16[8,65540]{{.*}} parameter(0)

// 1. Convert BF16 -> F32, then Bitcast to U32 and S32
// CHECK: %[[F32_VAL:[^ ]+]] = f32[8,65540]{{.*}} convert(%[[ARG]])
// CHECK: %[[VAL_U32:[^ ]+]] = u32[8,65540]{{.*}} bitcast-convert(%[[F32_VAL]])
// CHECK: %[[VAL_S32:[^ ]+]] = s32[8,65540]{{.*}} bitcast-convert(%[[F32_VAL]])

// 2. Pure Bitwise Radix Flip (F32 -> U32)
// CHECK: %[[SRA:[^ ]+]] = s32[8,65540]{{.*}} shift-right-arithmetic(%[[VAL_S32]], {{.*}})
// CHECK: %[[SRA_U32:[^ ]+]] = u32[8,65540]{{.*}} bitcast-convert(%[[SRA]])
// CHECK: %[[MASK:[^ ]+]] = u32[8,65540]{{.*}} or(%[[SRA_U32]], {{.*}})
// CHECK: %[[RADIX_KEY:[^ ]+]] = u32[8,65540]{{.*}} xor(%[[VAL_U32]], %[[MASK]])

// 3. Canonicalize -0.0f and pack into U64 with 31-bit reversed iota and is_neg_zero bit
// CHECK: %[[IS_NEG_ZERO_PRED:[^ ]+]] = pred[8,65540]{{.*}} compare(%[[VAL_U32]], {{.*}}), direction=EQ
// CHECK: %[[IS_NEG_ZERO:[^ ]+]] = u32[8,65540]{{.*}} convert(%[[IS_NEG_ZERO_PRED]])
// CHECK: %[[PARTIAL_KEY:[^ ]+]] = u32[8,65540]{{.*}} add(%[[RADIX_KEY]], %[[IS_NEG_ZERO]])
// CHECK: %[[KEY_U64:[^ ]+]] = u64[8,65540]{{.*}} convert(%[[PARTIAL_KEY]])
// CHECK: %[[SHIFT_LEFT:[^ ]+]] = u64[8,65540]{{.*}} shift-left(%[[KEY_U64]], {{.*}})
// CHECK: %[[IOTA:[^ ]+]] = u32[8,65540]{{.*}} iota(), iota_dimension=1
// CHECK: %[[IOTA_NEG:[^ ]+]] = u32[8,65540]{{.*}} subtract({{.*}}, %[[IOTA]])
// CHECK: %[[IOTA_SHIFTED:[^ ]+]] = u32[8,65540]{{.*}} shift-left(%[[IOTA_NEG]], {{.*}})
// CHECK: %[[LOW_U32:[^ ]+]] = u32[8,65540]{{.*}} or(%[[IOTA_SHIFTED]], %[[IS_NEG_ZERO]])
// CHECK: %[[LOW_U64:[^ ]+]] = u64[8,65540]{{.*}} convert(%[[LOW_U32]])
// CHECK: %[[PACKED:[^ ]+]] = u64[8,65540]{{.*}} or(%[[SHIFT_LEFT]], %[[LOW_U64]])

// 4. CustomCall (__gpu$$TopK)
// CHECK: %[[CUSTOM_CALL:[^ ]+]] = (u64[8,32]{{.*}}, s32[8,32]{{.*}}, u8[$0]{{.*}}) custom-call(%[[PACKED]]), custom_call_target="__gpu$$TopK", api_version=API_VERSION_TYPED_FFI, backend_config={is_stable = false, order = "PARTIAL"}

// 5. Unpack U64 -> U32 and restore -0.0f radix key
// CHECK: %[[SRL:[^ ]+]] = u64[8,32]{{.*}} shift-right-logical(%[[CUSTOM_CALL]]#0, {{.*}})
// CHECK: %[[PARTIAL_UNPACK_U32:[^ ]+]] = u32[8,32]{{.*}} convert(%[[SRL]])
// CHECK: %[[UNPACK_LOW_U32:[^ ]+]] = u32[8,32]{{.*}} convert(%[[CUSTOM_CALL]]#0)
// CHECK: %[[UNPACK_IS_NEG_ZERO:[^ ]+]] = u32[8,32]{{.*}} and(%[[UNPACK_LOW_U32]], {{.*}})
// CHECK: %[[UNPACK_U32:[^ ]+]] = u32[8,32]{{.*}} subtract(%[[PARTIAL_UNPACK_U32]], %[[UNPACK_IS_NEG_ZERO]])

// 6. Pure Bitwise Radix Unflip (U32 -> F32)
// CHECK: %[[MSB:[^ ]+]] = u32[8,32]{{.*}} shift-right-logical(%[[UNPACK_U32]], {{.*}})
// CHECK: %[[MSB_MINUS_ONE:[^ ]+]] = u32[8,32]{{.*}} subtract(%[[MSB]], {{.*}})
// CHECK: %[[UNMASK:[^ ]+]] = u32[8,32]{{.*}} or(%[[MSB_MINUS_ONE]], {{.*}})
// CHECK: %[[UNFLIPPED:[^ ]+]] = u32[8,32]{{.*}} xor(%[[UNPACK_U32]], %[[UNMASK]])
// CHECK: %[[F32_OUT:[^ ]+]] = f32[8,32]{{.*}} bitcast-convert(%[[UNFLIPPED]])

// 7. Convert F32 -> BF16 and Tuple output
// CHECK: %[[BF16_OUT:[^ ]+]] = bf16[8,32]{{.*}} convert(%[[F32_OUT]])
// CHECK: ROOT {{.*}} tuple({{.*}}%[[BF16_OUT]], {{.*}}%[[CUSTOM_CALL]]#1)
  )",
      expected_scratch_bytes);

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), check_pattern));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(TopkTest, SmallPartialOrderTopKUsesCustomKernel) {
  const char* hlo = R"(
    HloModule m

    %compare-gt.1 {
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      ROOT compare = pred[] compare(p.0.lhs, p.0.rhs), direction=GT
    }

    ENTRY top_k {
      arg = f32[8,1024] parameter(0)
      ROOT small_k = (f32[8,8], s32[8,8]) custom-call(arg), custom_call_target="TopK", called_computations={%compare-gt.1}, backend_config={is_stable = true, order = "PARTIAL"}
    }
  )";

  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options()
      .set_xla_gpu_experimental_enable_raft_for_stable_topk(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo, config));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       TopkSpecializer(device_description()).Run(module.get()));
  ASSERT_TRUE(changed);

  const char* check_pattern = R"(
// CHECK-LABEL: ENTRY %top_k
// CHECK: %[[ARG:[^ ]+]] = f32[8,1024]{{.*}} parameter(0)
// CHECK: %[[CUSTOM_CALL:[^ ]+]] = (f32[8,8]{{.*}}, s32[8,8]{{.*}}) custom-call(%[[ARG]]), custom_call_target="__gpu$TopK", api_version=API_VERSION_TYPED_FFI, backend_config={is_stable = true, order = "PARTIAL"}
// CHECK: ROOT {{.*}} tuple(%[[CUSTOM_CALL]]#0, %[[CUSTOM_CALL]]#1)
  )";

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), check_pattern));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(TopkTest, RewriteStablePartialOrderTopKProducesCorrectResult) {
  if (!device_description().gpu_compute_capability().IsCuda()) {
    GTEST_SKIP() << "RAFT is CUDA-only.";
  }

  for (int k : {8, 32}) {
    const std::string hlo = absl::Substitute(
        R"(
      HloModule m

      %compare-gt.1 {
        p.1.lhs = s32[] parameter(2)
        p.1.rhs = s32[] parameter(3)
        p.0.lhs = f32[] parameter(0)
        p.0.rhs = f32[] parameter(1)
        ROOT compare = pred[] compare(p.0.lhs, p.0.rhs), direction=GT
      }

      ENTRY top_k {
        arg = f32[4,1024] parameter(0)
        iota = s32[4,1024] iota(), iota_dimension=1
        c2 = s32[] constant(2)
        b2 = s32[4,1024] broadcast(c2), dimensions={}
        rem = s32[4,1024] remainder(iota, b2)
        c0 = s32[] constant(0)
        b0 = s32[4,1024] broadcast(c0), dimensions={}
        is_even = pred[4,1024] compare(rem, b0), direction=EQ
        pos_zero_c = f32[] constant(0)
        neg_zero_c = f32[] constant(-0)
        pos_zero = f32[4,1024] broadcast(pos_zero_c), dimensions={}
        neg_zero = f32[4,1024] broadcast(neg_zero_c), dimensions={}
        signed_zeros = f32[4,1024] select(is_even, pos_zero, neg_zero)
        ROOT result = (f32[4,$0], s32[4,$0]) custom-call(signed_zeros), custom_call_target="TopK", called_computations={%compare-gt.1}, backend_config={is_stable = true, order = "PARTIAL"}
      }
    )",
        k);

    HloModuleConfig config = GetModuleConfigForTest();
    config.mutable_debug_options()
        .set_xla_gpu_experimental_enable_raft_for_stable_topk(true);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> topk_module,
                         ParseAndReturnVerifiedModule(hlo, config));
    std::unique_ptr<HloModule> reference_module =
        topk_module->Clone("reference");

    ASSERT_OK_AND_ASSIGN(
        bool changed,
        TopkSpecializer(device_description()).Run(topk_module.get()));
    ASSERT_TRUE(changed);

    ASSERT_OK_AND_ASSIGN(changed, TopkDecomposer().Run(reference_module.get()));
    ASSERT_TRUE(changed);
    reference_module->mutable_config()
        .mutable_debug_options()
        .add_xla_disable_hlo_passes("topk-rewriter");

    EXPECT_TRUE(RunAndCompareTwoModules(
        std::move(topk_module), std::move(reference_module), std::nullopt));
  }
}

TEST_F(TopkTest, UnstableFloatTopKUsesEstimatedScratchSize) {
  if (!device_description().gpu_compute_capability().IsCuda()) {
    GTEST_SKIP() << "RAFT is CUDA-only.";
  }

  // Test both a WarpSort shape (f32[8,1024], k=64) and a Radix11bits shape
  // (f32[64,131072], k=256) to verify HLO scratch sizing and GPU execution.
  struct TestCase {
    int batch;
    int n;
    int k;
  };
  for (const TestCase& tc :
       {TestCase{8, 1024, 64}, TestCase{64, 131072, 256}}) {
    const std::string hlo = absl::Substitute(
        R"hlo(
      HloModule m

      %compare-gt.1 {
        p.1.lhs = s32[] parameter(2)
        p.1.rhs = s32[] parameter(3)
        p.0.lhs = f32[] parameter(0)
        p.0.rhs = f32[] parameter(1)
        ROOT compare = pred[] compare(p.0.lhs, p.0.rhs), direction=GT, order=TOTAL
      }

      ENTRY top_k {
        arg = f32[$0,$1] parameter(0)
        topk = (f32[$0,$2], s32[$0,$2]) custom-call(arg), custom_call_target="TopK", called_computations={%compare-gt.1}, backend_config={is_stable = false, order = "TOTAL"}
        ROOT values = f32[$0,$2] get-tuple-element(topk), index=0
      }
    )hlo",
        tc.batch, tc.n, tc.k);

    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> topk_module,
                         ParseAndReturnVerifiedModule(hlo));
    std::unique_ptr<HloModule> reference_module =
        topk_module->Clone("reference");

    ASSERT_OK_AND_ASSIGN(
        bool changed,
        TopkSpecializer(device_description()).Run(topk_module.get()));
    ASSERT_TRUE(changed);

    const se::CudaComputeCapability* cuda_cc =
        device_description().gpu_compute_capability().cuda_compute_capability();
    ASSERT_NE(cuda_cc, nullptr);
    const int expected_sm_count =
        GetRaftSelectKMaxSmCount(*cuda_cc, device_description().core_count());
    ASSERT_OK_AND_ASSIGN(int64_t expected_scratch_bytes,
                         EstimateRaftSelectKScratchBytesFloat(
                             tc.batch, tc.n, tc.k, expected_sm_count,
                             RaftScratchPhaseMode::kSumAcrossPhases));

    const HloInstruction* root =
        topk_module->entry_computation()->root_instruction();
    const HloInstruction* custom_call =
        root->operand(0)->operand(0)->operand(0);
    EXPECT_EQ(custom_call->custom_call_target(), "__gpu$TopK");
    ASSERT_EQ(custom_call->shape().tuple_shapes().size(), 3);
    EXPECT_EQ(custom_call->shape().tuple_shapes(2).dimensions(0),
              expected_scratch_bytes);

    ASSERT_OK_AND_ASSIGN(changed, TopkDecomposer().Run(reference_module.get()));
    ASSERT_TRUE(changed);
    reference_module->mutable_config()
        .mutable_debug_options()
        .add_xla_disable_hlo_passes("topk-rewriter");

    EXPECT_TRUE(RunAndCompareTwoModules(
        std::move(topk_module), std::move(reference_module), std::nullopt));
  }
}

TEST_F(TopkTest, UnstableBFloat16TopKUsesEstimatedScratchSize) {
  if (!device_description().gpu_compute_capability().IsCuda()) {
    GTEST_SKIP() << "RAFT is CUDA-only.";
  }

  // Test both a BF16 WarpSort shape (bf16[8,1024], k=32) and a BF16 Radix
  // shape (bf16[64,65536], k=256) to verify HLO scratch sizing and GPU
  // execution.
  struct TestCase {
    int batch;
    int n;
    int k;
  };
  for (const TestCase& tc : {TestCase{8, 1024, 32}, TestCase{64, 65536, 256}}) {
    const std::string hlo = absl::Substitute(
        R"hlo(
      HloModule m

      %compare-gt.1 {
        p.1.lhs = s32[] parameter(2)
        p.1.rhs = s32[] parameter(3)
        p.0.lhs = bf16[] parameter(0)
        p.0.rhs = bf16[] parameter(1)
        ROOT compare = pred[] compare(p.0.lhs, p.0.rhs), direction=GT, order=TOTAL
      }

      ENTRY top_k {
        arg = bf16[$0,$1] parameter(0)
        topk = (bf16[$0,$2], s32[$0,$2]) custom-call(arg), custom_call_target="TopK", called_computations={%compare-gt.1}, backend_config={is_stable = false, order = "TOTAL"}
        ROOT values = bf16[$0,$2] get-tuple-element(topk), index=0
      }
    )hlo",
        tc.batch, tc.n, tc.k);

    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> topk_module,
                         ParseAndReturnVerifiedModule(hlo));
    std::unique_ptr<HloModule> reference_module =
        topk_module->Clone("reference");

    ASSERT_OK_AND_ASSIGN(
        bool changed,
        TopkSpecializer(device_description()).Run(topk_module.get()));
    ASSERT_TRUE(changed);

    const se::CudaComputeCapability* cuda_cc =
        device_description().gpu_compute_capability().cuda_compute_capability();
    ASSERT_NE(cuda_cc, nullptr);
    const int expected_sm_count =
        GetRaftSelectKMaxSmCount(*cuda_cc, device_description().core_count());
    ASSERT_OK_AND_ASSIGN(int64_t expected_scratch_bytes,
                         EstimateRaftSelectKScratchBytesBFloat16(
                             tc.batch, tc.n, tc.k, expected_sm_count,
                             RaftScratchPhaseMode::kSumAcrossPhases));

    const HloInstruction* root =
        topk_module->entry_computation()->root_instruction();
    const HloInstruction* custom_call =
        root->operand(0)->operand(0)->operand(0);
    EXPECT_EQ(custom_call->custom_call_target(), "__gpu$TopK");
    ASSERT_EQ(custom_call->shape().tuple_shapes().size(), 3);
    EXPECT_EQ(custom_call->shape().tuple_shapes(2).dimensions(0),
              expected_scratch_bytes);

    ASSERT_OK_AND_ASSIGN(changed, TopkDecomposer().Run(reference_module.get()));
    ASSERT_TRUE(changed);
    reference_module->mutable_config()
        .mutable_debug_options()
        .add_xla_disable_hlo_passes("topk-rewriter");

    EXPECT_TRUE(RunAndCompareTwoModules(
        std::move(topk_module), std::move(reference_module), std::nullopt));
  }
}

TEST_F(TopkTest, StableTopKViaUint64UsesEstimatedScratchSize) {
  if (!device_description().gpu_compute_capability().IsCuda()) {
    GTEST_SKIP() << "RAFT is CUDA-only.";
  }

  // Verify HLO scratch sizing across WarpImmediate (f32[8,1024], k=32),
  // WarpFiltered (f32[64,16384], k=256), and Radix11bits (f32[64,32768],
  // k=256), and verify end-to-end GPU execution on the WarpSort shapes.
  struct TestCase {
    int batch;
    int n;
    int k;
  };
  // TODO(b/409009349): Add TestCase{64, 32768, 256} (Radix11bits) once
  // https://github.com/rapidsai/raft/pull/3212 is backported.
  for (const TestCase& tc : {TestCase{8, 1024, 32}, TestCase{64, 16384, 256}}) {
    const std::string hlo = absl::Substitute(
        R"hlo(
      HloModule m

      %compare-gt.1 {
        p.1.lhs = s32[] parameter(2)
        p.1.rhs = s32[] parameter(3)
        p.0.lhs = f32[] parameter(0)
        p.0.rhs = f32[] parameter(1)
        ROOT compare = pred[] compare(p.0.lhs, p.0.rhs), direction=GT, order=TOTAL
      }

      ENTRY top_k {
        arg = f32[$0,$1] parameter(0)
        ROOT topk = (f32[$0,$2], s32[$0,$2]) custom-call(arg), custom_call_target="TopK", called_computations={%compare-gt.1}, backend_config={is_stable = true, order = "TOTAL"}
      }
    )hlo",
        tc.batch, tc.n, tc.k);

    HloModuleConfig config = GetModuleConfigForTest();
    config.mutable_debug_options()
        .set_xla_gpu_experimental_enable_raft_for_stable_topk(true);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> topk_module,
                         ParseAndReturnVerifiedModule(hlo, config));
    std::unique_ptr<HloModule> reference_module =
        topk_module->Clone("reference");

    ASSERT_OK_AND_ASSIGN(
        bool changed,
        TopkSpecializer(device_description()).Run(topk_module.get()));
    ASSERT_TRUE(changed);

    const se::CudaComputeCapability* cuda_cc =
        device_description().gpu_compute_capability().cuda_compute_capability();
    ASSERT_NE(cuda_cc, nullptr);
    const int expected_sm_count =
        GetRaftSelectKMaxSmCount(*cuda_cc, device_description().core_count());
    ASSERT_OK_AND_ASSIGN(int64_t expected_scratch_bytes,
                         EstimateRaftSelectKScratchBytesUInt64(
                             tc.batch, tc.n, tc.k, expected_sm_count,
                             RaftScratchPhaseMode::kSumAcrossPhases));

    const HloInstruction* root =
        topk_module->entry_computation()->root_instruction();
    const HloInstruction* custom_call = root->operand(1)->operand(0);
    EXPECT_EQ(custom_call->custom_call_target(), "__gpu$TopK");
    ASSERT_EQ(custom_call->shape().tuple_shapes().size(), 3);
    EXPECT_EQ(custom_call->shape().tuple_shapes(0).element_type(), U64);
    EXPECT_EQ(custom_call->shape().tuple_shapes(2).dimensions(0),
              expected_scratch_bytes);

    ASSERT_OK_AND_ASSIGN(changed, TopkDecomposer().Run(reference_module.get()));
    ASSERT_TRUE(changed);
    reference_module->mutable_config()
        .mutable_debug_options()
        .add_xla_disable_hlo_passes("topk-rewriter");

    EXPECT_TRUE(RunAndCompareTwoModules(
        std::move(topk_module), std::move(reference_module), std::nullopt));
  }
}

TEST_F(TopkTest, RejectsZeroBatchStableTopKForRaftU64) {
  const char* hlo = R"hlo(
    HloModule m

    %compare-gt.1 {
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      ROOT compare = pred[] compare(p.0.lhs, p.0.rhs), direction=GT, order=TOTAL
    }

    ENTRY top_k {
      arg = f32[0,1024] parameter(0)
      ROOT result = (f32[0,32], s32[0,32]) custom-call(arg), custom_call_target="TopK", called_computations={%compare-gt.1}, backend_config={is_stable = true, order = "TOTAL"}
    }
  )hlo";

  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options()
      .set_xla_gpu_experimental_enable_raft_for_stable_topk(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo, config));

  if (!device_description().gpu_compute_capability().IsCuda()) {
    GTEST_SKIP() << "RAFT is CUDA-only.";
  }

  ASSERT_OK_AND_ASSIGN(bool changed,
                       TopkSpecializer(device_description()).Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(TopkTest, CrossSkuAndMigCompileExecutesSafelyOnFullGpu) {
  const se::CudaComputeCapability* cuda_cc =
      device_description().gpu_compute_capability().cuda_compute_capability();
  if (cuda_cc == nullptr) {
    GTEST_SKIP() << "RAFT is CUDA-only.";
  }

  // Simulate compiling on a reduced-capacity MIG partition (e.g. b200_mig = 18
  // SMs or h100_sxm_mig = 16 SMs) and executing the compiled module on a full
  // GPU. Without full-die SM upper-bounding, f32[1, 1048576] k=64 allocates
  // only ~280 KB for 18 SMs, whereas a full Hopper/Blackwell GPU launches ~4x
  // more WarpSort blocks and requires >= 1 MB of scratch.
  se::DeviceDescription mig_device_description = device_description();
  mig_device_description.set_core_count(18);

  const char* hlo = R"hlo(
    HloModule m

    %compare-gt.1 {
      p.1.lhs = s32[] parameter(2)
      p.1.rhs = s32[] parameter(3)
      p.0.lhs = f32[] parameter(0)
      p.0.rhs = f32[] parameter(1)
      ROOT compare = pred[] compare(p.0.lhs, p.0.rhs), direction=GT, order=TOTAL
    }

    ENTRY top_k {
      arg = f32[1,1048576] parameter(0)
      topk = (f32[1,64], s32[1,64]) custom-call(arg), custom_call_target="TopK", called_computations={%compare-gt.1}, backend_config={is_stable = false, order = "TOTAL"}
      ROOT values = f32[1,64] get-tuple-element(topk), index=0
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> topk_module,
                       ParseAndReturnVerifiedModule(hlo));
  std::unique_ptr<HloModule> reference_module = topk_module->Clone("reference");

  ASSERT_OK_AND_ASSIGN(
      bool changed,
      TopkSpecializer(mig_device_description).Run(topk_module.get()));
  ASSERT_TRUE(changed);

  ASSERT_OK_AND_ASSIGN(int64_t unsafe_mig_scratch_bytes,
                       EstimateRaftSelectKScratchBytesFloat(
                           1, 1048576, 64, /*sm_count=*/18,
                           RaftScratchPhaseMode::kSumAcrossPhases));
  ASSERT_OK_AND_ASSIGN(
      int64_t expected_full_die_scratch_bytes,
      EstimateRaftSelectKScratchBytesFloat(
          1, 1048576, 64,
          GetRaftSelectKMaxSmCount(*cuda_cc, /*reported_sm_count=*/18),
          RaftScratchPhaseMode::kSumAcrossPhases));
  if (device_description().core_count() > 18) {
    EXPECT_GT(expected_full_die_scratch_bytes, unsafe_mig_scratch_bytes);
  }

  const HloInstruction* root =
      topk_module->entry_computation()->root_instruction();
  const HloInstruction* custom_call = root->operand(0)->operand(0)->operand(0);
  EXPECT_EQ(custom_call->custom_call_target(), "__gpu$TopK");
  ASSERT_EQ(custom_call->shape().tuple_shapes().size(), 3);
  EXPECT_EQ(custom_call->shape().tuple_shapes(2).dimensions(0),
            expected_full_die_scratch_bytes);

  ASSERT_OK_AND_ASSIGN(changed, TopkDecomposer().Run(reference_module.get()));
  ASSERT_TRUE(changed);
  reference_module->mutable_config()
      .mutable_debug_options()
      .add_xla_disable_hlo_passes("topk-rewriter");

  EXPECT_TRUE(RunAndCompareTwoModules(
      std::move(topk_module), std::move(reference_module), std::nullopt));
}

TEST_F(TopkTest, RejectsTopKExceedingInt32DimensionLimit) {
  // n = 2^31 (INT32_MAX + 1) exceeds signed S32 index range and RAFT's 32-bit
  // signed int launch calculations, so TopkSpecializer must leave both unstable
  // and stable TopK unspecialized (for fallback decomposition).
  for (int k : {8, 32}) {
    for (absl::string_view is_stable : {"false", "true"}) {
      const std::string hlo = absl::Substitute(
          R"hlo(
        HloModule m

        %compare-gt.1 {
          p.1.lhs = s32[] parameter(2)
          p.1.rhs = s32[] parameter(3)
          p.0.lhs = f32[] parameter(0)
          p.0.rhs = f32[] parameter(1)
          ROOT compare = pred[] compare(p.0.lhs, p.0.rhs), direction=GT, order=TOTAL
        }

        ENTRY top_k {
          arg = f32[1,2147483648] parameter(0)
          ROOT result = (f32[1,$0], s32[1,$0]) custom-call(arg), custom_call_target="TopK", called_computations={%compare-gt.1}, backend_config={is_stable = $1, order = "TOTAL"}
        }
      )hlo",
          k, is_stable);

      HloModuleConfig config = GetModuleConfigForTest();
      config.mutable_debug_options()
          .set_xla_gpu_experimental_enable_raft_for_stable_topk(true);
      ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                           ParseAndReturnVerifiedModule(hlo, config));
      ASSERT_OK_AND_ASSIGN(
          bool changed,
          TopkSpecializer(device_description()).Run(module.get()));
      EXPECT_FALSE(changed) << "k=" << k << ", is_stable=" << is_stable;
    }
  }
}

}  // namespace
}  // namespace xla::gpu
