/* Copyright 2023 The OpenXLA Authors.

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
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/gpu/transforms/nest_gemm_fusion.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

class TritonTest : public GpuCodegenTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    // Do not fall back to cuBLAS, we are testing Triton.
    debug_options.set_xla_gpu_cublas_fallback(false);
    // Do not autotune split-k by default, since this prevents deterministically
    // matching the optimized HLO.
    debug_options.set_xla_gpu_enable_split_k_autotuning(false);
    // Always rewrite Gemms with Triton regardless of size.
    debug_options.set_xla_gpu_gemm_rewrite_size_threshold(0);
    debug_options
        .set_xla_gpu_experimental_enable_subchannel_dequantisation_fusion(true);
    return debug_options;
  }

  ::testing::AssertionResult RunAndCompare(absl::string_view hlo_text,
                                           ErrorSpec error_spec) {
    auto module_or = GetOptimizedModule(hlo_text);
    if (!module_or.ok()) {
      return ::testing::AssertionFailure() << module_or.status().message();
    }
    return RunAndCompareNoHloPasses(std::move(*module_or), error_spec);
  }

  ::testing::AssertionResult RunAndCompare(std::unique_ptr<HloModule> module,
                                           ErrorSpec error_spec) {
    auto module_or = GetOptimizedModule(std::move(module));
    if (!module_or.ok()) {
      return ::testing::AssertionFailure() << module_or.status().message();
    }
    return RunAndCompareNoHloPasses(std::move(*module_or), error_spec);
  }

  ::testing::AssertionResult RunAndCompareNoHloPasses(
      absl::string_view hlo_text, ErrorSpec error_spec) {
    auto module_or = ParseAndReturnVerifiedModule(hlo_text);
    if (!module_or.ok()) {
      return ::testing::AssertionFailure() << module_or.status().message();
    }
    return RunAndCompareNoHloPasses(std::move(*module_or), error_spec);
  }

  ::testing::AssertionResult RunAndCompareNoHloPasses(
      std::unique_ptr<HloModule> module, ErrorSpec error_spec) {
    if (auto status = MaybeAddTritonGemmConfig(module.get()); !status.ok()) {
      return ::testing::AssertionFailure() << status.message();
    }
    auto nested_or = NestGemmFusion(device_desc().gpu_compute_capability())
                         .Run(module.get());
    if (!nested_or.ok()) {
      return ::testing::AssertionFailure() << nested_or.status().message();
    }
    EXPECT_TRUE(nested_or.value());
    return GpuCodegenTest::RunAndCompareNoHloPasses(std::move(module),
                                                    error_spec);
  }

  absl::Status MaybeAddTritonGemmConfig(HloModule* module) {
    auto* fusion = Cast<HloFusionInstruction>(
        module->entry_computation()->root_instruction());
    if (!fusion) {
      return absl::InternalError("Entry root is not a fusion.");
    }
    TF_ASSIGN_OR_RETURN(auto gpu_config,
                        fusion->backend_config<GpuBackendConfig>());
    FusionBackendConfig* backend_config =
        gpu_config.mutable_fusion_backend_config();
    if (backend_config->has_triton_gemm_config()) {
      return absl::OkStatus();
    }
    auto* triton_gemm_key = backend_config->mutable_triton_gemm_config();
    triton_gemm_key->set_block_m(64);
    triton_gemm_key->set_block_k(64);
    triton_gemm_key->set_block_n(64);
    triton_gemm_key->set_split_k(1);
    triton_gemm_key->set_num_stages(1);
    triton_gemm_key->set_num_warps(2);
    triton_gemm_key->set_num_ctas(1);
    return fusion->set_backend_config(gpu_config);
  }

 protected:
  const stream_executor::DeviceDescription& device_desc() {
    return backend().default_stream_executor()->GetDeviceDescription();
  }
};

// The following tests are for the channel and subchannel dequantization
// fusions. We run the fused version to avoid the hlo passes and prove that
// emitters work correctly and unfused version with the goal to fail if an hlo
// rewrite broke the dequantization logic.
// For the subchannel dequantization there are two cases:
// 1. The case where we do:
//   broadcast -> multiply -> bitcast -> dot.
// 2. The case where we do:
//   broadcast -> reshape -> multiply -> dot.
// On top of that there could be an additional bitcast between the parameter and
// the broadcast.
TEST_F(TritonTest, FuseChannelDequantizationFused) {
  // This test is a Channel Dequantization fusion.
  // We run the fused version to avoid the hlo passes.
  // The case where we do:
  // param(1) -> bitcast -> broadcast -> multiply -> bitcast -> dot.
  constexpr absl::string_view kHloText = R"(
    HloModule FuseChannelDequantizationFused

    fusion {
      w.s4 = s4[32,128,256]{2,1,0:E(4)} parameter(0)
      w.s8 = s8[32,128,256] convert(w.s4)
      w.b16 = bf16[32,128,256] convert(w.s8)

      s = bf16[32,1,256] parameter(1)
      s.bitcast = bf16[32,256] bitcast(s)
      s.broadcast = bf16[32,128,256] broadcast(s.bitcast), dimensions={0,2}
      w.scaled = bf16[32,128,256] multiply(w.b16, s.broadcast)
      w.scaled.bitcast = bf16[32,2,64,256] bitcast(w.scaled)

      a = bf16[1,32,128,2,128] parameter(2)
      a.bitcast = bf16[32,128,256] bitcast(bf16[1,32,128,2,128] a)
      a.bitcast.2 = bf16[32,2,64,256] bitcast(a.bitcast)
      dot = f32[2,32,256,256] dot(w.scaled.bitcast, a.bitcast.2),
        lhs_batch_dims={1,0}, lhs_contracting_dims={2},
        rhs_batch_dims={1,0}, rhs_contracting_dims={2}
      ROOT bitcast = f32[2,32,256,2,1,128] bitcast(f32[2,32,256,256] dot)
    }

    ENTRY entry_computation {
      w.s4 = s4[32,128,256]{2,1,0:E(4)} parameter(0)
      s.bf16 = bf16[32,1,256] parameter(1)
      a.bf16 = bf16[1,32,128,2,128] parameter(2)
      ROOT fusion = f32[2,32,256,2,1,128] fusion(w.s4, s.bf16, a.bf16),
          kind=kCustom,
          calls=fusion,
          backend_config={
            "operation_queue_id":"0",
            "wait_on_operation_queues":[],
            "fusion_backend_config":{
              "kind":"__triton_gemm",
              "triton_gemm_config":{
                "block_m":"128",
                "block_n":"128",
                "block_k":"64",
                "split_k":"2",
                "num_stages":"1",
                "num_warps":"8",
                "num_ctas":"1"
              }
            },
            "force_earliest_schedule":false
          }
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, DISABLED_FuseSubchannelDequantizationWithTranspose) {
  constexpr absl::string_view kHloText = R"(
    HloModule FuseSubchannelDequantizationWithTranspose

    ENTRY FuseSubchannelDequantizationWithTranspose {
      w_s4 = s4[2,2048,64] parameter(1)
      w_s8 = s8[2,2048,64] convert(w_s4)
      w_s8_reshaped = s8[2,8,256,64] reshape(w_s8)
      w_bf16 = bf16[2,8,256,64] convert(w_s8_reshaped)
      s_bf16 = bf16[2,8,1,64]{3,1,0,2} parameter(0)
      s_bf16_reshaped = bf16[2,8,64] reshape(s_bf16)
      s_bf16_broadcasted = bf16[2,8,256,64] broadcast(s_bf16_reshaped),
          dimensions={0,1,3}
      w_bf16_scaled = bf16[2,8,256,64] multiply(w_bf16, s_bf16_broadcasted)
      w_bf16_scaled_reshaped = bf16[2,2048,64] reshape(w_bf16_scaled)

      a_bf16 = bf16[2,2048,2,32] parameter(2)
      a_bf16_reshaped = bf16[2,2048,64] reshape(a_bf16)
      dot = bf16[2,64,64] dot(w_bf16_scaled_reshaped, a_bf16_reshaped),
          lhs_batch_dims={0}, lhs_contracting_dims={1},
          rhs_batch_dims={0}, rhs_contracting_dims={1}
      dot_reshaped = bf16[2,64,2,32] reshape(dot)
      dot_transposed = bf16[64,2,2,32] transpose(dot_reshaped),
          dimensions={1,0,2,3}
      ROOT root = bf16[2,64,2,32]{3,2,0,1} reshape(dot_transposed)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    CHECK:    %[[bitcast:.*]] = bf16[2,8,64]{2,1,0} bitcast({{.*}})
    CHECK:    %[[transpose:.*]] = bf16[2,64,8]{2,1,0} transpose(%[[bitcast]]), dimensions={0,2,1}
    CHECK:    %[[broadcast:.*]] = bf16[2,64,8,256]{3,2,1,0} broadcast(%[[transpose]]), dimensions={0,1,2}
    CHECK:    %[[multiply:.*]] = bf16[2,64,8,256]{3,2,1,0} multiply({{.*}}, %[[broadcast]])
  )"));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), "CHECK: __triton_gemm"));

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(TritonTest, DISABLED_FuseSubchannelDequantization) {
  // This test is a Subchannel Dequantization fusion.
  // We run the non-fused version with the goal to fail if an hlo rewrite broke
  // the dequantization logic. The case where we do:
  //  param(1) -> reshape -> broadcast -> multiply -> reshape -> dot.
  constexpr absl::string_view kHloText = R"(
    HloModule FuseSubchannelDequantization

    ENTRY main {
      w = s4[2,2048,32] parameter(0)
      w.s8 = s8[2,2048,32] convert(w)
      w.b16 = bf16[2,2048,32] convert(w.s8)
      w.b16.reshaped = bf16[2,8,256,32] reshape(w.b16)

      s = bf16[2,8,1,32] parameter(1)
      s.reshaped = bf16[2,8,32] reshape(s)
      s.broadcasted = bf16[2,8,256,32] broadcast(s.reshaped),
          dimensions={0,1,3}
      w.scaled = bf16[2,8,256,32] multiply(w.b16.reshaped, s.broadcasted)
      w.scaled.reshaped = bf16[2,2048,32] reshape(w.scaled)

      a = bf16[2,2,1,2048] parameter(2)
      a.reshaped = bf16[2,2,2048] reshape(a)
      ROOT dot = f32[2,32,2] dot(w.scaled.reshaped, a.reshaped),
          lhs_batch_dims={0}, lhs_contracting_dims={1},
          rhs_batch_dims={1}, rhs_contracting_dims={2}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), "CHECK: __triton_gemm"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, FuseChannelDequantization) {
  // This test is a Channel Dequantization fusion.
  // We run the non-fused version with the goal to fail if an hlo rewrite broke
  // the dequantization logic. The case where we do:
  //  param(1) -> bitcast -> broadcast -> multiply -> dot.
  constexpr absl::string_view kHloText = R"(
    HloModule FuseChannelDequantization

    ENTRY main {
      w.s4 = s4[32,128,256] parameter(0)
      w.s8 = s8[32,128,256] convert(w.s4)
      w.bf16 = bf16[32,128,256] convert(w.s8)

      s = bf16[32,1,256] parameter(1)
      s.broadcast = bf16[32,1,256] broadcast(s), dimensions={0,1,2}
      s.reshape = bf16[32,256] reshape(s.broadcast)
      s.broadcast.2 = bf16[32,128,256] broadcast(s.reshape),
          dimensions={0,2}
      w.scaled = bf16[32,128,256] multiply(w.bf16, s.broadcast.2)

      a = bf16[2,1,32,128,128] parameter(2)
      ROOT dot = f32[32,256,2,1,128] dot(w.scaled, a),
          lhs_batch_dims={0}, lhs_contracting_dims={1},
          rhs_batch_dims={2}, rhs_contracting_dims={4}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), "CHECK: __triton_gemm"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, DISABLED_FuseSubchannelDequantizationFused) {
  // This test is a Subchannel Dequantization fusion.
  // We run the fused version to avoid the hlo passes.
  // The case where we do:
  // param -> bitcast -> broadcast -> multiply -> bitcast -> dot.
  constexpr absl::string_view kHloText = R"(
    HloModule FuseSubchannelDequantizationFused

    fusion {
      w.s4 = s4[2,2048,32]{2,1,0:E(4)} parameter(0)
      w.s8 = s8[2,2048,32] convert(w.s4)
      w.s8.bitcast = s8[2,8,256,32] bitcast(w.s8)
      w.bf16 = bf16[2,8,256,32] convert(w.s8.bitcast)

      s.bf16 = bf16[2,8,1,32] parameter(1)
      s.bf16.bitcast = bf16[2,8,32] bitcast(s.bf16)
      s.bf16.broadcast = bf16[2,8,256,32] broadcast(s.bf16.bitcast), dimensions={0,1,3}
      w = bf16[2,8,256,32] multiply(w.bf16, s.bf16.broadcast)
      w.bitcast = bf16[2,2048,32] bitcast(w)

      a = bf16[2,2,1,2048] parameter(2)
      a.bitcast = bf16[2,2,2048] bitcast(a)
      ROOT dot = f32[2,32,2] dot(w.bitcast, a.bitcast),
          lhs_batch_dims={0}, lhs_contracting_dims={1},
          rhs_batch_dims={1}, rhs_contracting_dims={2}
    } // fusion

    ENTRY main {
      w.s4 = s4[2,2048,32]{2,1,0:E(4)} parameter(0)
      s.bf16 = bf16[2,8,1,32] parameter(1)
      a.bf16 = bf16[2,2,1,2048] parameter(2)
      ROOT fusion = f32[2,32,2] fusion(w.s4, s.bf16, a.bf16),
        kind=kCustom,
        calls=fusion,
        backend_config={
          "operation_queue_id":"0",
          "wait_on_operation_queues":[],
          "fusion_backend_config":{
            "kind":"__triton_gemm",
            "triton_gemm_config":{
              "block_m":16,
              "block_n":16,
              "block_k":256,
              "split_k":1,
              "num_stages":1,
              "num_warps":2,
              "num_ctas":1
            }
          },
          "force_earliest_schedule":false
        }
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest,
       DISABLED_FuseSubchannelDequantizationFusedWithSmallBlockKSize) {
  // This test is a Subchannel Dequantization fusion.
  // We run the fused version to avoid the hlo passes.
  // The case where we do:
  // param -> bitcast -> broadcast -> multiply -> bitcast -> dot.
  constexpr absl::string_view kHloText = R"(
    HloModule FuseSubchannelDequantizationFusedWithSmallBlockKSize

    fusion {
      w.s4 = s4[2,2048,32]{2,1,0:E(4)} parameter(0)
      w.s8 = s8[2,2048,32] convert(w.s4)
      w.s8.bitcast = s8[2,8,256,32] bitcast(w.s8)
      w.bf16 = bf16[2,8,256,32] convert(w.s8.bitcast)

      s.bf16 = bf16[2,8,1,32] parameter(1)
      s.bf16.bitcast = bf16[2,8,32] bitcast(s.bf16)
      s.bf16.broadcast = bf16[2,8,256,32] broadcast(s.bf16.bitcast), dimensions={0,1,3}
      w = bf16[2,8,256,32] multiply(w.bf16, s.bf16.broadcast)
      w.bitcast = bf16[2,2048,32] bitcast(w)

      a = bf16[2,2,1,2048] parameter(2)
      a.bitcast = bf16[2,2,2048] bitcast(a)
      ROOT dot = f32[2,32,2] dot(w.bitcast, a.bitcast), 
          lhs_batch_dims={0}, lhs_contracting_dims={1},
          rhs_batch_dims={1}, rhs_contracting_dims={2}
    } // fusion

    ENTRY main {
      w.s4 = s4[2,2048,32]{2,1,0:E(4)} parameter(0)
      s.bf16 = bf16[2,8,1,32] parameter(1)
      a.bf16 = bf16[2,2,1,2048] parameter(2)
      ROOT fusion = f32[2,32,2] fusion(w.s4, s.bf16, a.bf16),
        kind=kCustom,
        calls=fusion,
        backend_config={
          "operation_queue_id":"0",
          "wait_on_operation_queues":[],
          "fusion_backend_config":{
            "kind":"__triton_gemm",
            "triton_gemm_config":{
              "block_m":16,
              "block_n":16,
              "block_k":128,
              "split_k":1,
              "num_stages":1,
              "num_warps":2,
              "num_ctas":1
            }
          },
          "force_earliest_schedule":false
        }
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, FuseBroadcastInPrologue) {
  constexpr absl::string_view kHloText = R"(
    HloModule FuseBroadcastInPrologue

    ENTRY main {
      lhs = bf16[2,1024] parameter(0)
      lhs.broadcast = bf16[2,256,1024] broadcast(lhs), dimensions={0,2}

      rhs = bf16[2,256,512] parameter(1)

      ROOT dot = f32[2,1024,512] dot(lhs.broadcast, rhs),
        lhs_batch_dims={0}, lhs_contracting_dims={1},
        rhs_batch_dims={0}, rhs_contracting_dims={1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    CHECK:    %[[broadcast:.*]] = bf16[2,256,1024]{2,1,0} broadcast
    CHECK:    %[[dot:.*]] = f32[2,1024,512]{2,1,0} dot
    CHECK:    ENTRY %main
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, DISABLED_FuseBroadcastBitcastInPrologue) {
  // This test is a Subchannel Dequantization fusion.
  constexpr absl::string_view kHloText = R"(
    HloModule FuseBroadcastBitcastInPrologue

    ENTRY main {
      lhs = bf16[2,1024] parameter(0)
      lhs.broadcast = bf16[2,128,1024] broadcast(lhs), dimensions={0,2}
      lhs.bitcast = bf16[256,1024] reshape(lhs.broadcast)

      rhs = bf16[256,512] parameter(1)

      ROOT dot = f32[1024,512] dot(lhs.bitcast, rhs),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    CHECK:    %[[broadcast:.*]] = bf16[{{.*}}]{2,1,0} broadcast
    CHECK:    %[[bitcast:.*]] = bf16[{{.*}}]{1,0} bitcast
    CHECK:    ROOT %[[dot:.*]] = f32[{{.*}}]{1,0} dot
    CHECK:    ENTRY %main
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}));
}

TEST_F(TritonTest, DISABLED_FuseBroadcastBitcastMultiplyInPrologue) {
  // This test is a Subchannel Dequantization fusion.
  constexpr absl::string_view kHloText = R"(
    HloModule FuseBroadcastBitcastMultiplyInPrologue

    ENTRY main {
      lhs = bf16[2,1024] parameter(0)
      lhs.broadcast = bf16[2,128,1024] broadcast(lhs), dimensions={0,2}
      lhs.bitcast = bf16[256,1024] reshape(lhs.broadcast)

      lhs.weights = s4[256,1024] parameter(1)
      lhs.weights.i8 = s8[256,1024] convert(lhs.weights)
      lhs.weights.bf16 = bf16[256,1024] convert(lhs.weights.i8)
      lhs.weights.scaled = bf16[256,1024] multiply(lhs.bitcast, lhs.weights.bf16)

      rhs = bf16[256,512] parameter(2)

      ROOT dot = f32[1024,512] dot(lhs.weights.scaled, rhs),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    CHECK:    %[[broadcast:.*]] = bf16[2,128,1024]{2,1,0} broadcast
    CHECK:    %[[bitcast:.*]] = bf16[256,1024]{1,0} bitcast
    CHECK:    %[[multiply:.*]] = [[type:.*]][256,1024]{1,0} multiply
    CHECK:    %[[dot:.*]] = f32[1024,512]{1,0} dot
    CHECK:    ENTRY %main
  )"));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}));
}

TEST_F(TritonTest, DotWithI4WeightsOnLhsWithBitcastTo3dTensor) {
  constexpr absl::string_view kHloText = R"(
    HloModule DotWithI4WeightsOnLhsWithBitcastTo3dTensor

    fusion {
      p0 = s4[256,16]{1,0:E(4)} parameter(0)
      p0.2 = bf16[256,16] convert(p0)
      p0.3 = bf16[4,64,16] bitcast(p0.2)
      p1 = bf16[4,32,64] parameter(1)
      ROOT dot = bf16[4,16,32] dot(p0.3, p1),
        lhs_batch_dims={0}, lhs_contracting_dims={1},
        rhs_batch_dims={0}, rhs_contracting_dims={2}
    }

    ENTRY entry_computation {
      p0 = s4[256,16]{1,0:E(4)} parameter(0)
      p1 = bf16[4,32,64] parameter(1)
      ROOT dot = bf16[4,16,32] fusion(p0, p1),
        kind=kCustom,
        calls=fusion,
        backend_config={
          "fusion_backend_config":{
            "kind":"__triton_gemm"
          }
        }
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}));
}

TEST_F(TritonTest,
       DotWithI4WeightsOnLhsWithNonStandardLayoutAndMultplyInEpilogue) {
  constexpr absl::string_view kHloText = R"(
    HloModule DotWithI4WeightsOnLhsWithNonStandardLayoutAndMultplyInEpilogue

    fusion {
      p0 = s4[1,128,32]{1,2,0:E(4)} parameter(0)
      p0.1 = s4[1,32,128]{2,1,0:E(4)} bitcast(p0)
      p0.2 = bf16[1,32,128] convert(p0.1)
      p0.3 = bf16[1,128,32]{1,2,0} bitcast(p0.2)
      p1 = bf16[128,1,64] parameter(1)
      dot = bf16[1,32,64] dot(p0.3, p1),
        lhs_batch_dims={0}, lhs_contracting_dims={1},
        rhs_batch_dims={1}, rhs_contracting_dims={0}
      p2 = bf16[1,1,32]{2,0,1} parameter(2)
      p2.1 = bf16[1,32] bitcast(p2)
      p2.2 = bf16[1,32,64] broadcast(p2.1), dimensions={0,1}
      m = bf16[1,32,64] multiply(dot, p2.2)
      ROOT m.1 = bf16[1,1,32,64] bitcast(m)
    }

    ENTRY entry_computation {
      p0 = s4[1,128,32]{1,2,0:E(4)} parameter(0)
      p1 = bf16[128,1,64] parameter(1)
      p2 = bf16[1,1,32]{2,0,1} parameter(2)
      ROOT gemm_fusion_dot.2 = bf16[1,1,32,64] fusion(p0, p1, p2),
        kind=kCustom,
        calls=fusion,
        backend_config={
          "fusion_backend_config":{
            "kind":"__triton_gemm"
          }
        }
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}));
}

TEST_F(TritonTest, DotWithInt4WeightsOnLhsFusedWithMultiplyByChannelScales) {
  constexpr absl::string_view kHloText = R"(
    HloModule DotWithI4WeightsOnLhsFusedWithMultiplyByChannelScales

    DotWithI4WeightsOnLhsFusedWithMultiplyByChannelScales {
      w = s4[32,64,128] parameter(0)
      w.i8 = s8[32,64,128] convert(w)
      w.bf16 = bf16[32,64,128] convert(w.i8)
      scales = bf16[32,128] parameter(1)
      scales.broadcast = bf16[32,64,128] broadcast(scales), dimensions={0,2}
      weights.scaled = bf16[32,64,128] multiply(w.bf16, scales.broadcast)
      activations = bf16[32,64,256] parameter(2)
      ROOT dot = f32[32,128,256] dot(weights.scaled, activations),
        lhs_batch_dims={0}, lhs_contracting_dims={1},
        rhs_batch_dims={0}, rhs_contracting_dims={1}
    }

    ENTRY main {
      w = s4[32,64,128] parameter(0)
      scales = bf16[32,128] parameter(1)
      p2 = bf16[32,64,256] parameter(2)
      ROOT dot = f32[32,128,256] fusion(w, scales, p2),
        kind=kCustom,
        calls=DotWithI4WeightsOnLhsFusedWithMultiplyByChannelScales,
        backend_config={
          "fusion_backend_config":{
            "kind":"__triton_gemm"
          }
        }
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}));
}

TEST_F(TritonTest, FuseMultiplyInPrologue) {
  constexpr absl::string_view kHloText = R"(
    HloModule FuseMultiplyInPrologue

    ENTRY main {
      t = (s4[32,64,128], bf16[32,128]{0,1}, bf16[32,64,256]) parameter(0)
      w = s4[32,64,128] get-tuple-element(t), index=0
      w.i8 = s8[32,64,128] convert(w)
      w.bf16 = bf16[32,64,128] convert(w.i8)
      scales = bf16[32,128]{0,1} get-tuple-element(t), index=1
      scales.broadcast = bf16[32,64,128] broadcast(scales), dimensions={0,2}
      weights.scaled = bf16[32,64,128] multiply(w.bf16, scales.broadcast)
      activations = bf16[32,64,256] get-tuple-element(t), index=2
      ROOT dot = f32[32,128,256] dot(weights.scaled, activations),
        lhs_batch_dims={0}, lhs_contracting_dims={1},
        rhs_batch_dims={0}, rhs_contracting_dims={1}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  // On Ampere the multiply result type is f32, on Hopper it is bf16.
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
    CHECK:    %[[multiply:.*]] = [[type:.*]]{{.*}} multiply({{.*}}, {{.*}})
    CHECK:    %[[dot:.*]] = f32[32,128,256]{2,1,0} dot
    CHECK:    ENTRY %main
  )"));
}

TEST_F(TritonTest, FuseMultiplyInEpilogue) {
  constexpr absl::string_view kHloText = R"(
    HloModule FuseMultiplyInEpilogue

    ENTRY main {
      p0 = s4[4,32,128]{2,1,0:E(4)} parameter(0)
      p0.1 = bf16[4,32,128] convert(p0)
      p1 = bf16[4,128,64] parameter(1)
      dot = bf16[4,32,64] dot(p0.1, p1),
        lhs_batch_dims={0}, lhs_contracting_dims={2},
        rhs_batch_dims={0}, rhs_contracting_dims={1}
      p2 = bf16[4,32] parameter(2)
      p2.1 = bf16[4,32,64] broadcast(p2), dimensions={0,1}
      ROOT m = bf16[4,32,64] multiply(dot, p2.1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
      CHECK:  %[[dot:.*]] = bf16[4,64,32]{1,2,0} dot
      CHECK:  %[[multiply:.*]] = [[type:.*]][4,32,64]{2,1,0} multiply
      CHECK:  ENTRY %main
    )"));
}

TEST_F(TritonTest, NonstandardLayoutInt4) {
  constexpr absl::string_view kHloText = R"(
    HloModule NonstandardLayoutInt4

    ENTRY main {
      p0 = s4[64,128]{0,1} parameter(0)
      p1 = bf16[256,64] parameter(1)
      ROOT dot = bf16[128,256] dot(p0, p1),
        lhs_contracting_dims={0}, rhs_contracting_dims={1}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

using ::testing::TestParamInfo;
using ::testing::WithParamInterface;

struct I4TestParams {
  static std::string ToString(const TestParamInfo<I4TestParams>& params) {
    return params.param.name;
  }

  std::string Format(absl::string_view format) const {
    return absl::StrReplaceAll(
        format, {{"${name}", name},
                 {"${lhs}", lhs},
                 {"${rhs}", rhs},
                 {"${lhs_contracting_dim}", absl::StrCat(lhs_contracting_dim)},
                 {"${rhs_contracting_dim}", absl::StrCat(rhs_contracting_dim)},
                 {"${out}", out}});
  }
  bool HasBatchDim() const {
    return std::vector<std::string>(absl::StrSplit(lhs, ',')).size() > 2;
  }

  std::string name;         // The name of the test.
  std::string lhs;          // The lhs shape like "128,16".
  std::string rhs;          // The rhs shape like "128,256".
  int lhs_contracting_dim;  // The contracting dimension of the lhs.
  int rhs_contracting_dim;  // The contracting dimension of the rhs.
  std::string out;          // The output shape like "16,256".
};

class ParametrizedTritonTest : public TritonTest,
                               public WithParamInterface<I4TestParams> {};

TEST_P(ParametrizedTritonTest, Int4WeightsOnTheLhs) {
  if (GetParam().HasBatchDim()) {
    GTEST_SKIP() << "2d test ignores batch dim case.";
  }
  constexpr absl::string_view kHloTextTemplate = R"(
    HloModule lhs_${name}

    lhs_${name} {
      w.s4 = s4[${lhs}] parameter(0)
      w.s8 = s8[${lhs}] convert(w.s4)
      w.bf16 = bf16[${lhs}] convert(w.s8)
      a = bf16[${rhs}] parameter(1)
      ROOT lhs_${name} = f32[${out}] dot(w.bf16, a),
        lhs_contracting_dims={${lhs_contracting_dim}},
        rhs_contracting_dims={${rhs_contracting_dim}}
    }

    ENTRY main {
      w = s4[${lhs}] parameter(0)
      a = bf16[${rhs}] parameter(1)
      ROOT gemm_fusion_dot.2 = f32[${out}] fusion(w, a),
        kind=kCustom,
        calls=lhs_${name},
        backend_config={
          "fusion_backend_config":{
            "kind":"__triton_gemm"
          }
        }
    }
  )";
  std::string hlo_text = GetParam().Format(kHloTextTemplate);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text,
                                       ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}))
      << "Failed for HLO: " << hlo_text;
}

TEST_P(ParametrizedTritonTest, Int4WeightsOnTheLhsWithBatchDim) {
  if (!GetParam().HasBatchDim()) {
    GTEST_SKIP() << "3d test ignores 2d case.";
  }
  constexpr absl::string_view kHloTextTemplate = R"(
    HloModule ${name}

    fusion {
      w.s4 = s4[${lhs}] parameter(0)
      w.s8 = s8[${lhs}] convert(w.s4)
      w.bf16 = bf16[${lhs}] convert(w.s8)
      a = bf16[${rhs}] parameter(1)
      ROOT dot.0 = f32[${out}] dot(w.bf16, a),
        lhs_batch_dims={0}, lhs_contracting_dims={${lhs_contracting_dim}},
        rhs_batch_dims={0}, rhs_contracting_dims={${rhs_contracting_dim}}
    }

    ENTRY gemm_fusion_dot_computation {
      w = s4[${lhs}] parameter(0)
      a = bf16[${rhs}] parameter(1)
      ROOT gemm_fusion_dot.2 = f32[${out}] fusion(w, a),
        kind=kCustom,
        calls=fusion,
        backend_config={
          "fusion_backend_config":{
            "kind":"__triton_gemm"
          }
        }
    }
  )";
  std::string hlo_text = GetParam().Format(kHloTextTemplate);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text,
                                       ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}))
      << "Failed for HLO: " << hlo_text;
}

TEST_P(ParametrizedTritonTest, Int4WeightsOnTheRhs) {
  if (GetParam().HasBatchDim()) {
    GTEST_SKIP() << "2d test ignores batch dim case.";
  }

  constexpr absl::string_view kHloTextTemplate = R"(
    HloModule rhs_${name}

    rhs_${name} {
      a = bf16[${lhs}] parameter(0)
      w.s4 = s4[${rhs}] parameter(1)
      w.s8 = s8[${rhs}] convert(w.s4)
      w.bf16 = bf16[${rhs}] convert(w.s8)
      ROOT rhs_${name} = f32[${out}] dot(a, w.bf16),
        lhs_contracting_dims={${lhs_contracting_dim}},
        rhs_contracting_dims={${rhs_contracting_dim}}
    }

    ENTRY main {
      a = bf16[${lhs}] parameter(0)
      w = s4[${rhs}] parameter(1)
      ROOT rhs_${name} = f32[${out}] fusion(a, w),
        kind=kCustom,
        calls=rhs_${name},
        backend_config={
          "fusion_backend_config":{
            "kind":"__triton_gemm"
          }
        }
    }
  )";
  std::string hlo_text = GetParam().Format(kHloTextTemplate);
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text,
                                       ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}))
      << "Failed for HLO: " << hlo_text;
}

std::vector<I4TestParams> Int4TestCases() {
  return {
      {"int4_dot_128_16_x_128_256", "128,16", "128,256", 0, 0, "16,256"},
      {"int4_dot_128_16_x_256_128", "128,16", "256,128", 0, 1, "16,256"},
      {"int4_dot_16_128_x_256_128", "16,128", "256,128", 1, 1, "16,256"},
      {"int4_dot_16_128_x_128_256", "16,128", "128,256", 1, 0, "16,256"},
      // These fail for Int4WeightsOnTheRhs only.
      {"DISABLED_int4_dot_1_128_x_256_128", "1,128", "256,128", 1, 1, "1,256"},
      {"DISABLED_int4_dot_128_1_x_256_128", "128,1", "256,128", 0, 1, "1,256"},
      // These fail for Int4WeightsOnTheLhs only.
      {"DISABLED_int4_dot_16_128_x_128_1", "16,128", "128,1", 1, 0, "16,1"},
      {"DISABLED_int4_dot_16_128_x_1_128", "16,128", "1,128", 1, 1, "16,1"},

      {"dot_8_128_16_x_8_128_256", "8,128,16", "8,128,256", 1, 1, "8,16,256"},
      {"dot_8_128_16_x_8_256_128", "8,128,16", "8,256,128", 1, 2, "8,16,256"},
      {"dot_8_16_128_x_8_256_128", "8,16,128", "8,256,128", 2, 2, "8,16,256"},
      {"dot_8_16_128_x_8_128_256", "8,16,128", "8,128,256", 2, 1, "8,16,256"},
      {"DISABLED_dot_8_1_128_x_8_256_128", "8,1,128", "8,256,128", 2, 2,
       "8,1,256"},
      {"DISABLED_dot_8_128_1_x_8_256_128", "8,128,1", "8,256,128", 1, 2,
       "8,1,256"},
      {"dot_8_16_128_x_8_128_1", "8,16,128", "8,128,1", 2, 1, "8,16,1"},
      {"dot_8_16_128_x_8_1_128", "8,16,128", "8,1,128", 2, 2, "8,16,1"},
  };
}

INSTANTIATE_TEST_SUITE_P(ParametrizedTritonTest, ParametrizedTritonTest,
                         ::testing::ValuesIn(Int4TestCases()),
                         I4TestParams::ToString);

TEST_F(TritonTest, NonstandardLayoutWithManyNonContractingDims) {
  constexpr absl::string_view kHloText = R"(
    HloModule NonstandardLayoutWithManyNonContractingDims

    ENTRY main {
          p0 = s4[128,64,192]{1,0,2} parameter(0)
          p1 = bf16[256,64] parameter(1)
          ROOT dot = bf16[128,192,256] dot(p0, p1),
            lhs_contracting_dims={1}, rhs_contracting_dims={1}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-2}));
}

TEST_F(TritonTest, NonstandardLayoutWithManyNonContractingDimsReversedLayout) {
  // We cannot do triton_gemm and we use cuBLAS instead.
  constexpr absl::string_view kHloText = R"(
    HloModule NonstandardLayoutWithManyNonContractingDimsReversedLayout

    ENTRY main {
          lhs = s4[128,64,192]{0,1,2} parameter(0)
          rhs = bf16[256,64] parameter(1)
          ROOT dot = bf16[128,192,256] dot(lhs, rhs),
            lhs_contracting_dims={1}, rhs_contracting_dims={1}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      std::move(module), ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, NegatePlusConvertHLO) {
  constexpr absl::string_view kHloText = R"(
    HloModule NegatePlusConvertHLO

    ENTRY main {
      lhs = s4[2,32,64] parameter(0)
      lhs_negated = s4[2,32,64] negate(lhs)
      lhs_converted = bf16[2,32,64] convert(lhs_negated)
      rhs = bf16[2,64,16] parameter(1)
      ROOT dot = bf16[2,32,16] dot(lhs_converted, rhs),
          lhs_batch_dims={0}, lhs_contracting_dims={2},
          rhs_batch_dims={0}, rhs_contracting_dims={1}
    }
  )";
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, RejectTritonFusionForWithMinorBatchDim) {
  constexpr absl::string_view kHloText = R"(
    HloModule RejectTritonFusionForWithMinorBatchDim

    ENTRY main {
      lhs = s4[32,64,2] parameter(0)
      lhs_converted = bf16[32,64,2] convert(lhs)
      rhs = bf16[2,64,16] parameter(1)
      ROOT dot = bf16[2,32,16] dot(lhs_converted, rhs),
          lhs_batch_dims={2}, lhs_contracting_dims={1},
          rhs_batch_dims={0}, rhs_contracting_dims={1}
    }
  )";

  const std::string pattern =
      R"(CHECK-NOT: "kind":"__triton_gemm","triton_gemm_config")";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), pattern));
}

TEST_F(TritonTest, DISABLED_LHSWithMinorDimEqualTo1) {
  // We prove that triton can handle int4 dot with non contracting dim size
  // equal to 1.
  constexpr absl::string_view kHloText = R"(
    HloModule LHSWithMinorDimEqualTo1

    triton_computation {
      lhs = s4[2,1024,1] parameter(0)
      lhs_converted = bf16[2,1024,1] convert(lhs)
      rhs = bf16[2,64,1024] parameter(1)
      ROOT dot = bf16[2,1,64] dot(lhs_converted, rhs),
          lhs_batch_dims={0}, lhs_contracting_dims={1},
          rhs_batch_dims={0}, rhs_contracting_dims={2}
    }

    ENTRY main {
      lhs = s4[2,1024,1] parameter(0)
      rhs = bf16[2,64,1024] parameter(1)
      ROOT dot = bf16[2,1,64] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, DISABLED_RHSWithMinorDimEqualTo1) {
  // We prove that triton can handle int4 dot with non contracting dim size
  // equal to 1.
  constexpr absl::string_view kHloText = R"(
    HloModule RHSWithMinorDimEqualTo1

    triton_computation {
      lhs = bf16[2,1024,64] parameter(0)
      rhs = s4[2,1024,1] parameter(1)
      rhs_converted = bf16[2,1024,1] convert(rhs)
      ROOT dot = bf16[2,64,1] dot(lhs, rhs_converted),
          lhs_batch_dims={0}, lhs_contracting_dims={1},
          rhs_batch_dims={0}, rhs_contracting_dims={1}
    }

    ENTRY main {
      lhs = bf16[2,1024,64] parameter(0)
      rhs = s4[2,1024,1] parameter(1)
      ROOT dot = bf16[2,64,1] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, DISABLED_LHSNonMinorContractingDim) {
  // We prove that triton can handle int4 dot with non minor
  // lhs_contracting_dim.
  constexpr absl::string_view kHloText = R"(
    HloModule LHSNonMinorContractingDim

    triton_computation {
      lhs = s4[1024,8] parameter(0)
      lhs_converted = bf16[1024,8] convert(lhs)
      rhs = bf16[1024,4] parameter(1)
      ROOT dot = bf16[8,4] dot(lhs_converted, rhs),
          lhs_contracting_dims={0}, rhs_contracting_dims={0}
    }

    ENTRY main {
      lhs = s4[1024,8] parameter(0)
      rhs = bf16[1024,4] parameter(1)
      ROOT dot = bf16[8,4] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, LHSNonMinorContractingDimWithBatchDim0) {
  // We prove that triton can handle int4 dot with non minor
  // lhs_contracting_dim.
  constexpr absl::string_view kHloText = R"(
    HloModule LHSNonMinorContractingDimWithBatchDim0

    triton_computation {
      lhs = s4[2,1024,8] parameter(0)
      lhs_converted = bf16[2,1024,8] convert(lhs)
      rhs = bf16[2,1024,4] parameter(1)
      ROOT dot = bf16[2,8,4] dot(lhs_converted, rhs),
        lhs_batch_dims={0}, lhs_contracting_dims={1},
        rhs_batch_dims={0}, rhs_contracting_dims={1}
    }

    ENTRY main {
      lhs = s4[2,1024,8] parameter(0)
      rhs = bf16[2,1024,4] parameter(1)
      ROOT dot = bf16[2,8,4] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, LHSMinorContractingDim) {
  // We prove that triton can handle int4 dot with minor lhs_contracting_dim.
  constexpr absl::string_view kHloText = R"(
    HloModule LHSMinorContractingDim

    triton_computation {
      lhs = s4[8,1024] parameter(0)
      lhs_converted = bf16[8,1024] convert(lhs)
      rhs = bf16[1024,4] parameter(1)
      ROOT dot = bf16[8,4] dot(lhs_converted, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY main {
      lhs = s4[8,1024] parameter(0)
      rhs = bf16[1024,4] parameter(1)
      ROOT dot = bf16[8,4] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(TritonTest, ConvertPlusNegate) {
  constexpr absl::string_view kHloText = R"(
    HloModule ConvertPlusNegate

    triton_computation {
      lhs = s4[8,1024] parameter(0)
      lhs_converted = bf16[8,1024] convert(lhs)
      lhs_negated = bf16[8,1024] negate(lhs_converted)
      rhs = bf16[1024,4] parameter(1)
      ROOT dot = bf16[8,4] dot(lhs_negated, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY main {
      lhs = s4[8,1024] parameter(0)
      rhs = bf16[1024,4] parameter(1)
      ROOT dot = bf16[8,4] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(TritonTest, LHSMinorContractingDimWithBatchDim0) {
  // We prove that triton can handle int4 dot with minor lhs_contracting_dim.
  constexpr absl::string_view kHloText = R"(
    HloModule LHSMinorContractingDimWithBatchDim0

    triton_computation {
      lhs = s4[2,8,1024] parameter(0)
      lhs_converted = bf16[2,8,1024] convert(lhs)
      rhs = bf16[2,1024,4] parameter(1)
      ROOT dot = bf16[2,8,4] dot(lhs_converted, rhs),
        lhs_batch_dims={0}, lhs_contracting_dims={2},
        rhs_batch_dims={0}, rhs_contracting_dims={1}
    }

    ENTRY main {
      lhs = s4[2,8,1024] parameter(0)
      rhs = bf16[2,1024,4] parameter(1)
      ROOT dot = bf16[2,8,4] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(TritonTest, RHSTestWithNotMinorContractingDim) {
  constexpr absl::string_view kHloText = R"(
    HloModule RHSTestWithNotMinorContractingDim

    triton_computation {
      lhs = bf16[8,1024] parameter(0)
      rhs = s4[1024,4] parameter(1)
      rhs_converted = bf16[1024,4] convert(rhs)
      ROOT dot = bf16[8,4] dot(lhs, rhs_converted),
          lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY main {
      lhs = bf16[8,1024] parameter(0)
      rhs = s4[1024,4] parameter(1)
      ROOT dot = bf16[8,4] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(TritonTest, RHSTestWithMinorContractingDim) {
  constexpr absl::string_view kHloText = R"(
    HloModule RHSTestWithMinorContractingDim

    triton_computation {
      lhs = bf16[8,1024] parameter(0)
      rhs = s4[4,1024] parameter(1)
      rhs_converted = bf16[4,1024] convert(rhs)
      ROOT dot = bf16[8,4] dot(lhs, rhs_converted),
          lhs_contracting_dims={1}, rhs_contracting_dims={1}
    }

    ENTRY main {
      lhs = bf16[8,1024] parameter(0)
      rhs = s4[4,1024] parameter(1)
      ROOT dot = bf16[8,4] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(TritonTest, RHSTestWithMinorContractingDimWithBatchDim) {
  constexpr absl::string_view kHloText = R"(
    HloModule RHSTestWithMinorContractingDimWithBatchDim

    triton_computation {
      lhs = bf16[2,8,1024] parameter(0)
      rhs = s4[2,1024,4] parameter(1)
      rhs_converted = bf16[2,1024,4] convert(rhs)
      ROOT dot = bf16[2,8,4] dot(lhs, rhs_converted),
          lhs_batch_dims={0}, lhs_contracting_dims={2},
          rhs_batch_dims={0}, rhs_contracting_dims={1}
    }

    ENTRY main {
      lhs = bf16[2,8,1024] parameter(0)
      rhs = s4[2,1024,4] parameter(1)
      ROOT dot = bf16[2,8,4] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

TEST_F(TritonTest, RHSTestWithNotMinorContractingDimWithBatchDim0) {
  constexpr absl::string_view kHloText = R"(
    HloModule RHSTestWithNotMinorContractingDimWithBatchDim0

    triton_computation {
      lhs = bf16[2,8,1024] parameter(0)
      rhs = s4[2,4,1024] parameter(1)
      rhs_converted = bf16[2,4,1024] convert(rhs)
      ROOT dot = bf16[2,8,4] dot(lhs, rhs_converted),
          lhs_batch_dims={0}, lhs_contracting_dims={2},
          rhs_batch_dims={0}, rhs_contracting_dims={2}
    }

    ENTRY main {
      lhs = bf16[2,8,1024] parameter(0)
      rhs = s4[2,4,1024] parameter(1)
      ROOT dot = bf16[2,8,4] fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-2}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
