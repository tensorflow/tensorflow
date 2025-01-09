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

#include <gtest/gtest.h>
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

class TritonInt4Test : public GpuCodegenTest {
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

TEST_F(TritonInt4Test, NonstandardLayout) {
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

TEST_F(TritonInt4Test, NonstandardLayoutWithManyNonContractingDims) {
  // We cannot do triton_gemm and we use cuBLAS instead.
  constexpr absl::string_view kHloText = R"(
    HloModule t

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

TEST_F(TritonInt4Test,
       NonstandardLayoutWithManyNonContractingDimsReversedLayout) {
  // We cannot do triton_gemm and we use cuBLAS instead.
  constexpr absl::string_view kHloText = R"(
    HloModule t

    ENTRY main {
          p0 = s4[128,64,192]{0,1,2} parameter(0)
          p1 = bf16[256,64]{1,0} parameter(1)
          ROOT %dot = bf16[128,192,256]{2,1,0} dot(p0, p1),
            lhs_contracting_dims={1},
            rhs_contracting_dims={1}
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module, GetOptimizedModule(kHloText));
  EXPECT_TRUE(*RunFileCheck(module->ToString(), R"(CHECK:  "__cublas$gemm")"));
  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonInt4Test, NegatePlusConvertHLO) {
  constexpr absl::string_view kHloText = R"(
    HloModule t

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

TEST_F(TritonInt4Test, RejectTritonFusionForWithMinorBatchDim) {
  constexpr absl::string_view kHloText = R"(
    HloModule t

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

TEST_F(TritonInt4Test, LHSWithMinorDimEqualTo1) {
  // We prove that triton can handle int4 dot with non contracting dim size
  // equal to 1.
  constexpr absl::string_view kHloText = R"(
    HloModule t

    triton_computation {
      lhs = s4[16,32,1]{2,1,0} parameter(0)
      lhs_converted = bf16[16,32,1]{2,1,0} convert(lhs)
      rhs = bf16[16,64,32]{2,1,0} parameter(1)
      ROOT dot = bf16[16,1,64]{2,1,0} dot(lhs_converted, rhs),
          lhs_contracting_dims={1},
          rhs_contracting_dims={2},
          lhs_batch_dims={0},
          rhs_batch_dims={0}
    }

    ENTRY main {
      lhs = s4[16,32,1]{2,1,0} parameter(0)
      rhs = bf16[16,64,32]{2,1,0} parameter(1)
      ROOT dot = bf16[16,1,64]{2,1,0} fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";
  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonInt4Test, RHSWithMinorDimEqualTo1) {
  // We prove that triton can handle int4 dot with non contracting dim size
  // equal to 1.
  constexpr absl::string_view kHloText = R"(
    HloModule t

    triton_computation {
      lhs = bf16[16,32,64]{2,1,0} parameter(0)
      rhs = s4[16,32,1]{2,1,0} parameter(1)
      rhs_converted = bf16[16,32,1]{2,1,0} convert(rhs)
      ROOT dot = bf16[16,64,1]{2,1,0} dot(lhs, rhs_converted),
          lhs_contracting_dims={1},
          rhs_contracting_dims={1},
          lhs_batch_dims={0},
          rhs_batch_dims={0}
    }

    ENTRY main {
      lhs = bf16[16,32,64]{2,1,0} parameter(0)
      rhs = s4[16,32,1]{2,1,0} parameter(1)
      ROOT dot = bf16[16,64,1]{2,1,0} fusion(lhs, rhs), kind=kCustom,
        calls=triton_computation,
        backend_config={"fusion_backend_config": {"kind":"__triton_gemm"}}
    }
  )";

  EXPECT_TRUE(RunAndCompareNoHloPasses(
      kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(TritonInt4Test, LHSNonMinorContractingDim) {
  // We prove that triton can handle int4 dot with non minor
  // lhs_contracting_dim.
  constexpr absl::string_view kHloText = R"(
    HloModule t

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

TEST_F(TritonInt4Test, LHSNonMinorContractingDimWithBatchDim0) {
  // We prove that triton can handle int4 dot with non minor
  // lhs_contracting_dim.
  constexpr absl::string_view kHloText = R"(
    HloModule t

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

TEST_F(TritonInt4Test, LHSMinorContractingDim) {
  // We prove that triton can handle int4 dot with minor lhs_contracting_dim.
  constexpr absl::string_view kHloText = R"(
    HloModule t

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

TEST_F(TritonInt4Test, ConvertPlusNegate) {
  constexpr absl::string_view kHloText = R"(
    HloModule t

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

TEST_F(TritonInt4Test, LHSMinorContractingDimWithBatchDim0) {
  // We prove that triton can handle int4 dot with minor lhs_contracting_dim.
  constexpr absl::string_view kHloText = R"(
    HloModule t

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

TEST_F(TritonInt4Test, RHSTestWithMinorContractingDim) {
  constexpr absl::string_view kHloText = R"(
    HloModule t

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

TEST_F(TritonInt4Test, RHSTestWithNotMinorContractingDim) {
  constexpr absl::string_view kHloText = R"(
    HloModule t

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

TEST_F(TritonInt4Test, RHSTestWithMinorContractingDimWithBatchDim) {
  constexpr absl::string_view kHloText = R"(
    HloModule t

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

TEST_F(TritonInt4Test, RHSTestWithNotMinorContractingDimWithBatchDim0) {
  constexpr absl::string_view kHloText = R"(
    HloModule t

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
