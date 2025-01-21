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
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
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
    return debug_options;
  }

  stream_executor::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }

  const stream_executor::GpuComputeCapability& GpuComputeComp() {
    return device_desc().gpu_compute_capability();
  }
  stream_executor::GpuComputeCapability CudaAmpereOrRocm() {
    if (std::holds_alternative<stream_executor::RocmComputeCapability>(
            GpuComputeComp())) {
      return stream_executor::GpuComputeCapability{
          device_desc().rocm_compute_capability()};
    } else {
      return stream_executor::GpuComputeCapability{
          stream_executor::CudaComputeCapability{
              stream_executor::CudaComputeCapability::AMPERE, 0}};
    }
  }

 protected:
  const stream_executor::DeviceDescription& device_desc() {
    return backend().default_stream_executor()->GetDeviceDescription();
  }
};

TEST_F(TritonTest, DotWithI4WeightsOnLhsWithBitcastTo3dTensor) {
  constexpr absl::string_view kHloText = R"(
    HloModule DotWithI4WeightsOnLhsWithBitcastTo3dTensor

<<<<<<< HEAD:third_party/xla/xla/service/gpu/fusions/triton/triton_fusion_emitter_int4_device_test.cc
TEST_F(PlainInt4ToPackedInt4RewritePassTest,
       DotWithI4WeightsOnLhsFusedWithMultiplyByChannelScales) {
  GTEST_SKIP() << "TODO(rocm): Weekly-sync 25-01-13: Skip ivestigate int4 "
                  "issue with triton.";
=======
    fusion {
      p_0 = s4[256,16]{1,0:E(4)} parameter(0)
      p_0.2 = bf16[256,16]{1,0} convert(p_0)
      p_0.3 = bf16[4,64,16]{2,1,0} bitcast(p_0.2)
      p_1 = bf16[4,32,64]{2,1,0} parameter(1)
      ROOT dot = bf16[4,16,32]{2,1,0} dot(p_0.3, p_1),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={2}
    }

    ENTRY %entry_computation {
      p_0 = s4[256,16]{1,0:E(4)} parameter(0)
      p_1 = bf16[4,32,64]{2,1,0} parameter(1)
      ROOT dot = bf16[4,16,32]{2,1,0} fusion(p_0, p_1),
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
      p_0 = s4[1,128,32]{1,2,0:E(4)} parameter(0)
      p_0.1 = s4[1,32,128]{2,1,0:E(4)} bitcast(p_0)
      p_0.2 = bf16[1,32,128]{2,1,0} convert(p_0.1)
      p_0.3 = bf16[1,128,32]{1,2,0} bitcast(p_0.2)
      p_1 = bf16[128,1,64]{2,1,0} parameter(1)
      dot = bf16[1,32,64]{2,1,0} dot(p_0.3, p_1),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={1},
        rhs_contracting_dims={0}
      p_2 = bf16[1,1,32]{2,0,1} parameter(2)
      p_2.1 = bf16[1,32]{1,0} bitcast(p_2)
      p_2.2 = bf16[1,32,64]{2,1,0} broadcast(p_2.1), dimensions={0,1}
      m = bf16[1,32,64]{2,1,0} multiply(dot, p_2.2)
      ROOT m.1 = bf16[1,1,32,64]{3,2,1,0} bitcast(m)
    }

    ENTRY %entry_computation {
      p_0 = s4[1,128,32]{1,2,0:E(4)} parameter(0)
      p_1 = bf16[128,1,64]{2,1,0} parameter(1)
      p_2 = bf16[1,1,32]{2,0,1} parameter(2)
      ROOT gemm_fusion_dot.2 = bf16[1,1,32,64]{3,2,1,0} fusion(p_0, p_1, p_2),
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
>>>>>>> upstream/master:third_party/xla/xla/backends/gpu/codegen/triton/fusion_emitter_int4_device_test.cc
  constexpr absl::string_view kHloText = R"(
    HloModule DotWithI4WeightsOnLhsFusedWithMultiplyByChannelScales

    DotWithI4WeightsOnLhsFusedWithMultiplyByChannelScales {
      w = s4[32,64,128]{2,1,0} parameter(0)
      w.i8 = s8[32,64,128]{2,1,0} convert(w)
      w.f32 = f32[32,64,128]{2,1,0} convert(w.i8)
      scales = f32[32,128]{1,0} parameter(1)
      scales.broadcast = f32[32,64,128]{2,1,0} broadcast(scales), dimensions={0,2}
      weights.scaled = f32[32,64,128]{2,1,0} multiply(w.f32, scales.broadcast)
      activations = f32[32,64,256]{2,1,0} parameter(2)
      ROOT dot = f32[32,128,256]{2,1,0} dot(weights.scaled, activations),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1}
    }

    ENTRY main {
      w = s4[32,64,128]{2,1,0} parameter(0)
      scales = f32[32,128]{1,0} parameter(1)
      p2 = f32[32,64,256]{2,1,0} parameter(2)
      ROOT dot = f32[32,128,256]{2,1,0} fusion(w, scales, p2),
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

TEST_F(TritonTest, NonstandardLayoutInt4) {
  constexpr absl::string_view kHloText = R"(
    HloModule NonstandardLayoutInt4

    ENTRY main {
      p0 = s4[64,128]{0,1} parameter(0)
      p1 = bf16[256,64]{1,0} parameter(1)
      ROOT %dot = bf16[128,256]{1,0} dot(s4[64,128]{0,1} p0, bf16[256,64]{1,0} p1),
        lhs_contracting_dims={0},
        rhs_contracting_dims={1}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
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

<<<<<<< HEAD:third_party/xla/xla/service/gpu/fusions/triton/triton_fusion_emitter_int4_device_test.cc
TEST_P(ParametrizedPlainInt4ToPackedInt4RewritePassTest, Int4WeightsOnTheLhs) {
  GTEST_SKIP() << "TODO(rocm): Weekly-sync 25-01-13: Skip ivestigate int4 "
                  "issue with triton.";
=======
TEST_P(ParametrizedTritonTest, Int4WeightsOnTheLhs) {
>>>>>>> upstream/master:third_party/xla/xla/backends/gpu/codegen/triton/fusion_emitter_int4_device_test.cc
  if (GetParam().HasBatchDim()) {
    GTEST_SKIP() << "2d test ignores batch dim case.";
  }
  constexpr absl::string_view kHloTextTemplate = R"(
    HloModule lhs_${name}

    lhs_${name} {
      w.s4 = s4[${lhs}]{1,0} parameter(0)
      w.s8 = s8[${lhs}]{1,0} convert(w.s4)
      w.f32 = f32[${lhs}]{1,0} convert(w.s8)
      a = f32[${rhs}]{1,0} parameter(1)
      ROOT lhs_${name} = f32[${out}]{1,0} dot(w.f32, a),
        lhs_contracting_dims={${lhs_contracting_dim}},
        rhs_contracting_dims={${rhs_contracting_dim}}
    }

    ENTRY main {
      w = s4[${lhs}]{1,0} parameter(0)
      a = f32[${rhs}]{1,0} parameter(1)
      ROOT gemm_fusion_dot.2 = f32[${out}]{1,0} fusion(w, a),
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

<<<<<<< HEAD:third_party/xla/xla/service/gpu/fusions/triton/triton_fusion_emitter_int4_device_test.cc
TEST_P(ParametrizedPlainInt4ToPackedInt4RewritePassTest,
       Int4WeightsOnTheLhsWithBatchDim) {
  GTEST_SKIP() << "TODO(rocm): Weekly-sync 25-01-13: Skip ivestigate int4 "
                  "issue with triton.";
=======
TEST_P(ParametrizedTritonTest, Int4WeightsOnTheLhsWithBatchDim) {
>>>>>>> upstream/master:third_party/xla/xla/backends/gpu/codegen/triton/fusion_emitter_int4_device_test.cc
  if (!GetParam().HasBatchDim()) {
    GTEST_SKIP() << "3d test ignores 2d case.";
  }
  constexpr absl::string_view kHloTextTemplate = R"(
    HloModule ${name}

    fusion {
      w.s4 = s4[${lhs}]{2,1,0} parameter(0)
      w.s8 = s8[${lhs}]{2,1,0} convert(w.s4)
      w.f32 = f32[${lhs}]{2,1,0} convert(w.s8)
      a = f32[${rhs}]{2,1,0} parameter(1)
      ROOT dot.0 = f32[${out}]{2,1,0} dot(w.f32, a),
        lhs_contracting_dims={${lhs_contracting_dim}},
        rhs_contracting_dims={${rhs_contracting_dim}},
        lhs_batch_dims={0},
        rhs_batch_dims={0}
    }

    ENTRY gemm_fusion_dot_computation {
      w = s4[${lhs}]{2,1,0} parameter(0)
      a = f32[${rhs}]{2,1,0} parameter(1)
      ROOT gemm_fusion_dot.2 = f32[${out}]{2,1,0} fusion(w, a),
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

<<<<<<< HEAD:third_party/xla/xla/service/gpu/fusions/triton/triton_fusion_emitter_int4_device_test.cc
TEST_P(ParametrizedPlainInt4ToPackedInt4RewritePassTest, Int4WeightsOnTheRhs) {
  GTEST_SKIP()
      << "TODO: Weekly-sync 25-01-13: Skip ivestigate int4 issue with triton.";
=======
TEST_P(ParametrizedTritonTest, Int4WeightsOnTheRhs) {
>>>>>>> upstream/master:third_party/xla/xla/backends/gpu/codegen/triton/fusion_emitter_int4_device_test.cc
  if (GetParam().HasBatchDim()) {
    GTEST_SKIP() << "2d test ignores batch dim case.";
  }

  constexpr absl::string_view kHloTextTemplate = R"(
    HloModule rhs_${name}

    rhs_${name} {
      a = f32[${lhs}]{1,0} parameter(0)
      w.s4 = s4[${rhs}]{1,0} parameter(1)
      w.s8 = s8[${rhs}]{1,0} convert(w.s4)
      w.f32 = f32[${rhs}]{1,0} convert(w.s8)
      ROOT rhs_${name} = f32[${out}]{1,0} dot(a, w.f32),
        lhs_contracting_dims={${lhs_contracting_dim}},
        rhs_contracting_dims={${rhs_contracting_dim}}
    }

    ENTRY main {
      a = f32[${lhs}]{1,0} parameter(0)
      w = s4[${rhs}]{1,0} parameter(1)
      ROOT rhs_${name} = f32[${out}]{1,0} fusion(a, w),
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
      {"int4_dot_1_128_x_256_128", "1,128", "256,128", 1, 1, "1,256"},
      {"int4_dot_128_1_x_256_128", "128,1", "256,128", 0, 1, "1,256"},
      {"int4_dot_16_128_x_128_1", "16,128", "128,1", 1, 0, "16,1"},
      {"int4_dot_16_128_x_1_128", "16,128", "1,128", 1, 1, "16,1"},

      {"dot_8_128_16_x_8_128_256", "8,128,16", "8,128,256", 1, 1, "8,16,256"},
      {"dot_8_128_16_x_8_256_128", "8,128,16", "8,256,128", 1, 2, "8,16,256"},
      {"dot_8_16_128_x_8_256_128", "8,16,128", "8,256,128", 2, 2, "8,16,256"},
      {"dot_8_16_128_x_8_128_256", "8,16,128", "8,128,256", 2, 1, "8,16,256"},
      {"dot_8_1_128_x_8_256_128", "8,1,128", "8,256,128", 2, 2, "8,1,256"},
      {"dot_8_128_1_x_8_256_128", "8,128,1", "8,256,128", 1, 2, "8,1,256"},
      {"dot_8_16_128_x_8_128_1", "8,16,128", "8,128,1", 2, 1, "8,16,1"},
      {"dot_8_16_128_x_8_1_128", "8,16,128", "8,1,128", 2, 2, "8,16,1"},
  };
}

INSTANTIATE_TEST_SUITE_P(ParametrizedTritonTest, ParametrizedTritonTest,
                         ::testing::ValuesIn(Int4TestCases()),
                         I4TestParams::ToString);

<<<<<<< HEAD:third_party/xla/xla/service/gpu/fusions/triton/triton_fusion_emitter_int4_device_test.cc
TEST_F(TritonTest, NonstandardLayoutInt4) {
  GTEST_SKIP() << "TODO(rocm): Weekly-sync 25-01-13: Skip ivestigate int4 "
                  "issue with triton.";
  constexpr absl::string_view kHloText = R"(
    HloModule NonstandardLayout

    ENTRY main {
      p0 = s4[64,128]{0,1} parameter(0)
      p1 = bf16[256,64]{1,0} parameter(1)
      ROOT %dot = bf16[128,256]{1,0} dot(s4[64,128]{0,1} p0, bf16[256,64]{1,0} p1),
        lhs_contracting_dims={0},
        rhs_contracting_dims={1}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(
           CHECK:  %[[param_0:.*]] = s4[64,128]{0,1:E(4)} parameter(0)
           CHECK:  %[[bitcast:.*]] = s4[128,64]{1,0:E(4)} bitcast(s4[64,128]{0,1:E(4)} %[[param_0]])
           CHECK:  %[[convert:.*]] = bf16[128,64]{1,0} convert(s4[128,64]{1,0:E(4)} %[[bitcast]])
           CHECK:  %[[param_1:.*]] = bf16[256,64]{1,0} parameter(1)
           CHECK:  ROOT %dot.1 = bf16[128,256]{1,0} dot(bf16[128,64]{1,0} %[[convert]], bf16[256,64]{1,0} %[[param_1]]), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  )"));
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

=======
>>>>>>> upstream/master:third_party/xla/xla/backends/gpu/codegen/triton/fusion_emitter_int4_device_test.cc
TEST_F(TritonTest, NonstandardLayoutWithManyNonContractingDims) {
  GTEST_SKIP() << "TODO(rocm): Weekly-sync 25-01-13: Skip ivestigate int4 "
                  "issue with triton.";
  // We cannot do triton_gemm and we use cuBLAS instead.
  constexpr absl::string_view kHloText = R"(
    HloModule NonstandardLayoutWithManyNonContractingDims

    ENTRY main {
          p0 = s4[128,64,192]{1,0,2} parameter(0)
          p1 = bf16[256,64]{1,0} parameter(1)
          ROOT %dot = bf16[128,192,256]{2,1,0} dot(p0, p1),
            lhs_contracting_dims={1},
            rhs_contracting_dims={1}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(CHECK:  "__cublas$gemm")"));
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-2}));
}

TEST_F(TritonTest, NonstandardLayoutWithManyNonContractingDimsReversedLayout) {
  GTEST_SKIP() << "TODO(rocm): Weekly-sync 25-01-13: Skip ivestigate int4 "
                  "issue with triton.";
  // We cannot do triton_gemm and we use cuBLAS instead.
  constexpr absl::string_view kHloText = R"(
    HloModule NonstandardLayoutWithManyNonContractingDimsReversedLayout

    ENTRY main {
          lhs = s4[128,64,192]{0,1,2} parameter(0)
          rhs = bf16[256,64]{1,0} parameter(1)
          ROOT %dot = bf16[128,192,256]{2,1,0} dot(lhs, rhs),
            lhs_contracting_dims={1},
            rhs_contracting_dims={1}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(CHECK:  "__cublas$gemm")"));
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, NegatePlusConvertHLO) {
  constexpr absl::string_view kHloText = R"(
    HloModule NegatePlusConvertHLO

    ENTRY main {
      lhs = s4[16,32,64]{2,1,0} parameter(0)
      lhs_negated = s4[16,32,64]{2,1,0} negate(lhs)
      lhs_converted = bf16[16,32,64]{2,1,0} convert(lhs_negated)
      rhs = bf16[16,64,16]{2,1,0} parameter(1)
      ROOT dot = bf16[16,32,16]{2,1,0} dot(lhs_converted, rhs),
          lhs_contracting_dims={2},
          rhs_contracting_dims={1},
          lhs_batch_dims={0},
          rhs_batch_dims={0}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, RejectTritonFusionForWithMinorBatchDim) {
  constexpr absl::string_view kHloText = R"(
    HloModule RejectTritonFusionForWithMinorBatchDim

    ENTRY main {
      lhs = s4[32,64,16]{2,1,0} parameter(0)
      lhs_converted = bf16[32,64,16]{2,1,0} convert(lhs)
      rhs = bf16[16,64,16]{2,1,0} parameter(1)
      ROOT dot = bf16[16,32,16]{2,1,0} dot(lhs_converted, rhs),
          lhs_contracting_dims={1},
          rhs_contracting_dims={1},
          lhs_batch_dims={2},
          rhs_batch_dims={0}
    }
  )";

  const std::string pattern =
      R"(CHECK-NOT: "kind":"__triton_gemm","triton_gemm_config")";
  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  TF_ASSERT_OK_AND_ASSIGN(auto ok, RunFileCheck(module->ToString(), pattern));
  EXPECT_TRUE(ok);
}

TEST_F(TritonTest, LHSWithMinorDimEqualTo1) {
  // We prove that triton can handle int4 dot with non contracting dim size
  // equal to 1.
  constexpr absl::string_view kHloText = R"(
    HloModule LHSWithMinorDimEqualTo1

    triton_computation {
      lhs = s4[16,1024,1]{2,1,0} parameter(0)
      lhs_converted = bf16[16,1024,1]{2,1,0} convert(lhs)
      rhs = bf16[16,64,1024]{2,1,0} parameter(1)
      ROOT dot = bf16[16,1,64]{2,1,0} dot(lhs_converted, rhs),
          lhs_contracting_dims={1},
          rhs_contracting_dims={2},
          lhs_batch_dims={0},
          rhs_batch_dims={0}
    }

    ENTRY main {
      lhs = s4[16,1024,1]{2,1,0} parameter(0)
      rhs = bf16[16,64,1024]{2,1,0} parameter(1)
      ROOT dot = bf16[16,1,64]{2,1,0} fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, RHSWithMinorDimEqualTo1) {
  // We prove that triton can handle int4 dot with non contracting dim size
  // equal to 1.
  constexpr absl::string_view kHloText = R"(
    HloModule RHSWithMinorDimEqualTo1

    triton_computation {
      lhs = bf16[16,1024,64]{2,1,0} parameter(0)
      rhs = s4[16,1024,1]{2,1,0} parameter(1)
      rhs_converted = bf16[16,1024,1]{2,1,0} convert(rhs)
      ROOT dot = bf16[16,64,1]{2,1,0} dot(lhs, rhs_converted),
          lhs_contracting_dims={1},
          rhs_contracting_dims={1},
          lhs_batch_dims={0},
          rhs_batch_dims={0}
    }

    ENTRY main {
      lhs = bf16[16,1024,64]{2,1,0} parameter(0)
      rhs = s4[16,1024,1]{2,1,0} parameter(1)
      ROOT dot = bf16[16,64,1]{2,1,0} fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonTest, LHSNonMinorContractingDim) {
  // We prove that triton can handle int4 dot with non minor
  // lhs_contracting_dim.
  constexpr absl::string_view kHloText = R"(
    HloModule LHSNonMinorContractingDim

    triton_computation {
      lhs = s4[1024,8]{1,0} parameter(0)
      lhs_converted = bf16[1024,8]{1,0} convert(lhs)
      rhs = bf16[1024,4]{1,0} parameter(1)
      ROOT dot = bf16[8,4]{1,0} dot(lhs_converted, rhs),
          lhs_contracting_dims={0},
          rhs_contracting_dims={0}
    }

    ENTRY main {
      lhs = s4[1024,8]{1,0} parameter(0)
      rhs = bf16[1024,4]{1,0} parameter(1)
      ROOT dot = bf16[8,4]{1,0} fusion(lhs, rhs), kind=kCustom,
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
      lhs = s4[16,1024,8]{2,1,0} parameter(0)
      lhs_converted = bf16[16,1024,8]{2,1,0} convert(lhs)
      rhs = bf16[16,1024,4]{2,1,0} parameter(1)
      ROOT dot = bf16[16,8,4]{2,1,0} dot(lhs_converted, rhs),
        lhs_batch_dims={0},
        lhs_contracting_dims={1},
        rhs_batch_dims={0},
        rhs_contracting_dims={1}
    }

    ENTRY main {
      lhs = s4[16,1024,8]{2,1,0} parameter(0)
      rhs = bf16[16,1024,4]{2,1,0} parameter(1)
      ROOT dot = bf16[16,8,4]{2,1,0} fusion(lhs, rhs), kind=kCustom,
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
      lhs = s4[8,1024]{1,0} parameter(0)
      lhs_converted = bf16[8,1024]{1,0} convert(lhs)
      rhs = bf16[1024,4]{1,0} parameter(1)
      ROOT dot = bf16[8,4]{1,0} dot(lhs_converted, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY main {
      lhs = s4[8,1024]{1,0} parameter(0)
      rhs = bf16[1024,4]{1,0} parameter(1)
      ROOT dot = bf16[8,4]{1,0} fusion(lhs, rhs), kind=kCustom,
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
      lhs = s4[8,1024]{1,0} parameter(0)
      lhs_converted = bf16[8,1024]{1,0} convert(lhs)
      lhs_negated = bf16[8,1024]{1,0} negate(lhs_converted)
      rhs = bf16[1024,4]{1,0} parameter(1)
      ROOT dot = bf16[8,4]{1,0} dot(lhs_negated, rhs),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }

    ENTRY main {
      lhs = s4[8,1024]{1,0} parameter(0)
      rhs = bf16[1024,4]{1,0} parameter(1)
      ROOT dot = bf16[8,4]{1,0} fusion(lhs, rhs), kind=kCustom,
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
      lhs = s4[16,8,1024]{2,1,0} parameter(0)
      lhs_converted = bf16[16,8,1024]{2,1,0} convert(lhs)
      rhs = bf16[16,1024,4]{2,1,0} parameter(1)
      ROOT dot = bf16[16,8,4]{2,1,0} dot(lhs_converted, rhs),
        lhs_batch_dims={0},
        lhs_contracting_dims={2},
        rhs_batch_dims={0},
        rhs_contracting_dims={1}
    }

    ENTRY main {
      lhs = s4[16,8,1024]{2,1,0} parameter(0)
      rhs = bf16[16,1024,4]{2,1,0} parameter(1)
      ROOT dot = bf16[16,8,4]{2,1,0} fusion(lhs, rhs), kind=kCustom,
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
      lhs = bf16[8,1024]{1,0} parameter(0)
      rhs = s4[1024,4]{1,0} parameter(1)
      rhs_converted = bf16[1024,4]{1,0} convert(rhs)
      ROOT dot = bf16[8,4] dot(lhs, rhs_converted),
          lhs_contracting_dims={1},
          rhs_contracting_dims={0}
    }

    ENTRY main {
      lhs = bf16[8,1024]{1,0} parameter(0)
      rhs = s4[1024,4]{1,0} parameter(1)
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
      lhs = bf16[8,1024]{1,0} parameter(0)
      rhs = s4[4,1024]{1,0} parameter(1)
      rhs_converted = bf16[4,1024]{1,0} convert(rhs)
      ROOT dot = bf16[8,4] dot(lhs, rhs_converted),
          lhs_contracting_dims={1},
          rhs_contracting_dims={1}
    }

    ENTRY main {
      lhs = bf16[8,1024]{1,0} parameter(0)
      rhs = s4[4,1024]{1,0} parameter(1)
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
      lhs = bf16[16,8,1024]{2,1,0} parameter(0)
      rhs = s4[16,1024,4]{2,1,0} parameter(1)
      rhs_converted = bf16[16,1024,4]{2,1,0} convert(rhs)
      ROOT dot = bf16[16,8,4] dot(lhs, rhs_converted),
          lhs_batch_dims={0},
          lhs_contracting_dims={2},
          rhs_batch_dims={0},
          rhs_contracting_dims={1}
    }

    ENTRY main {
      lhs = bf16[16,8,1024]{2,1,0} parameter(0)
      rhs = s4[16,1024,4]{2,1,0} parameter(1)
      ROOT dot = bf16[16,8,4] fusion(lhs, rhs), kind=kCustom,
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
      lhs = bf16[16,8,1024]{2,1,0} parameter(0)
      rhs = s4[16,4,1024]{2,1,0} parameter(1)
      rhs_converted = bf16[16,4,1024]{2,1,0} convert(rhs)
      ROOT dot = bf16[16,8,4] dot(lhs, rhs_converted),
          lhs_batch_dims={0},
          lhs_contracting_dims={2},
          rhs_batch_dims={0},
          rhs_contracting_dims={2}
    }

    ENTRY main {
      lhs = bf16[16,8,1024]{2,1,0} parameter(0)
      rhs = s4[16,4,1024]{2,1,0} parameter(1)
      ROOT dot = bf16[16,8,4] fusion(lhs, rhs), kind=kCustom,
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
