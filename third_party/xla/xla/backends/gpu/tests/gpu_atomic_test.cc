/* Copyright 2018 The OpenXLA Authors.

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

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/types.h"

namespace xla {
namespace gpu {
namespace {

using GpuAtomicTest = GpuPjRtCodegenTest;

TEST_F(GpuAtomicTest, TestStore) {
  const char* hlo_string = R"(
    HloModule TensorFlowScatterV1

    update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
      lhs = s32[] parameter(0)
      ROOT rhs = s32[] parameter(1)
    }

    ENTRY main {
      operand = s32[3,3] parameter(0)
      indices = s32[2] parameter(1)
      updates = s32[2,3] parameter(2)
      ROOT scatter = s32[3,3] scatter(operand, indices, updates),
          to_apply=update_s32,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    }
)";

  ASSERT_OK(CompileAndVerifyIr(hlo_string, IsBuiltWithRocm() ? R"(
CHECK: store atomic{{.*}}syncscope("agent-one-as") monotonic, align 4
)"
                                                             : R"(
CHECK: store atomic{{.*}}unordered, align 4
)"));
}

TEST_F(GpuAtomicTest, TestStoreNoAtomic) {
  const char* hlo_string = R"(
    HloModule TensorFlowScatterV1

    update_s32 (lhs: s32[], rhs: s32[]) -> s32[] {
      lhs = s32[] parameter(0)
      ROOT rhs = s32[] parameter(1)
    }

    ENTRY main {
      operand = s32[3,3] parameter(0)
      indices = s32[2] parameter(1)
      updates = s32[2,3] parameter(2)
      ROOT scatter = s32[3,3] scatter(operand, indices, updates),
          to_apply=update_s32,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1, unique_indices=true
    }
)";

  ASSERT_OK(CompileAndVerifyIr(hlo_string, R"(
CHECK-NOT: store atomic
)"));
}

TEST_F(GpuAtomicTest, TestAddAtomicF32) {
  const char* hlo_string = R"(
    HloModule TensorFlowScatterV1

    update_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY main {
      operand = f32[3,3] parameter(0)
      indices = s32[2] parameter(1)
      updates = f32[2,3] parameter(2)
      ROOT scatter = f32[3,3] scatter(operand, indices, updates),
          to_apply=update_f32,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1, unique_indices=false
    }
)";

  ASSERT_OK(CompileAndVerifyIr(hlo_string, IsBuiltWithRocm() ? R"(
CHECK: atomicrmw fadd ptr %[[ADDR:.*]], float %[[VALUE:.*]] syncscope("agent-one-as") monotonic{{.*}}!atomic.ignore.denormal.mode
)"
                                                             : R"(
CHECK: atomicrmw fadd ptr %[[ADDR:.*]], float %[[VALUE:.*]] monotonic{{.*}}!atomic.ignore.denormal.mode
)"));
}

TEST_F(GpuAtomicTest, TestAddAtomicF16) {
  if (!IsBuiltWithRocm() &&
      !device_description().cuda_compute_capability().SupportsAllFeaturesOf(
          stream_executor::CudaComputeCapability::Volta())) {
    return;
  }
  const char* hlo_string = R"(
    HloModule TensorFlowScatterV1

    update_f16 (lhs: f16[], rhs: f16[]) -> f16[] {
      lhs = f16[] parameter(0)
      rhs = f16[] parameter(1)
      ROOT add = f16[] add(lhs, rhs)
    }

    ENTRY main {
      operand = f16[3,3] parameter(0)
      indices = s32[2] parameter(1)
      updates = f16[2,3] parameter(2)
      ROOT scatter = f16[3,3] scatter(operand, indices, updates),
          to_apply=update_f16,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1, unique_indices=false
    }
)";

  // ROCm GPUs with packed fp16 atomics add a <2 x half> to a global pointer.
  ASSERT_OK(CompileAndVerifyIr(hlo_string, IsBuiltWithRocm() ? R"(
CHECK: atomicrmw fadd ptr {{.*}}, {{(<2 x half>|half)}} %[[VALUE:.*]] syncscope("agent-one-as") monotonic{{.*}}!atomic.ignore.denormal.mode
)"
                                                             : R"(
CHECK: atomicrmw fadd ptr %[[ADDR:.*]], half %[[VALUE:.*]] monotonic{{.*}}!atomic.ignore.denormal.mode
)"));
}

TEST_F(GpuAtomicTest, TestAddAtomicBF16) {
  if (!IsBuiltWithRocm() &&
      !device_description().cuda_compute_capability().SupportsAllFeaturesOf(
          stream_executor::CudaComputeCapability::Hopper())) {
    return;
  }
  const char* hlo_string = R"(
    HloModule TensorFlowScatterV1

    update_bf16 (lhs: bf16[], rhs: bf16[]) -> bf16[] {
      lhs = bf16[] parameter(0)
      rhs = bf16[] parameter(1)
      ROOT add = bf16[] add(lhs, rhs)
    }

    ENTRY main {
      operand = bf16[3,3] parameter(0)
      indices = s32[2] parameter(1)
      updates = bf16[2,3] parameter(2)
      ROOT scatter = bf16[3,3] scatter(operand, indices, updates),
          to_apply=update_bf16,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1, unique_indices=false
    }
)";

  // ROCm GPUs with packed bf16 atomics add a <2 x bfloat> to a global pointer.
  ASSERT_OK(CompileAndVerifyIr(hlo_string, IsBuiltWithRocm() ? R"(
CHECK: atomicrmw fadd ptr {{.*}}, {{(<2 x bfloat>|bfloat)}} %[[VALUE:.*]] syncscope("agent-one-as") monotonic{{.*}}!atomic.ignore.denormal.mode
)"
                                                             : R"(
CHECK: atomicrmw fadd ptr %[[ADDR:.*]], bfloat %[[VALUE:.*]] monotonic{{.*}}!atomic.ignore.denormal.mode
)"));
}

TEST_F(GpuAtomicTest, TestAddAtomicF64) {
  // Atomic add required sm_60 or above.
  if (!IsBuiltWithRocm() &&
      !device_description().cuda_compute_capability().SupportsAllFeaturesOf(
          stream_executor::CudaComputeCapability::Pascal())) {
    return;
  }

  const char* hlo_string = R"(
    HloModule TensorFlowScatterV1

    update_f64 (lhs: f64[], rhs: f64[]) -> f64[] {
      lhs = f64[] parameter(0)
      rhs = f64[] parameter(1)
      ROOT add = f64[] add(lhs, rhs)
    }

    ENTRY main {
      operand = f64[3,3] parameter(0)
      indices = s32[2] parameter(1)
      updates = f64[2,3] parameter(2)
      ROOT scatter = f64[3,3] scatter(operand, indices, updates),
          to_apply=update_f64,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1, unique_indices=false
    }
)";

  ASSERT_OK(CompileAndVerifyIr(hlo_string, IsBuiltWithRocm() ? R"(
CHECK: atomicrmw fadd ptr %[[ADDR:.*]], double %[[VALUE:.*]] syncscope("agent-one-as") monotonic{{.*}}!atomic.ignore.denormal.mode
)"
                                                             : R"(
CHECK: atomicrmw fadd ptr %[[ADDR:.*]], double %[[VALUE:.*]] monotonic{{.*}}!atomic.ignore.denormal.mode
)"));
}

// complex<f32> only has the ABI alignment of f32, so the 64 bit atomic word
// relies on the alignment of the kernel argument that holds the operand.
TEST_F(GpuAtomicTest, TestStoreC64) {
  const char* hlo_string = R"(
    HloModule TensorFlowScatterV1

    update_c64 (lhs: c64[], rhs: c64[]) -> c64[] {
      lhs = c64[] parameter(0)
      ROOT rhs = c64[] parameter(1)
    }

    ENTRY main {
      operand = c64[3,3] parameter(0)
      indices = s32[2] parameter(1)
      updates = c64[2,3] parameter(2)
      ROOT scatter = c64[3,3] scatter(operand, indices, updates),
          to_apply=update_c64,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1, unique_indices=false
    }
)";

  ASSERT_OK(CompileAndVerifyIr(hlo_string, IsBuiltWithRocm() ? R"(
CHECK: store atomic i64 {{.*}}syncscope("agent-one-as") monotonic, align 8
)"
                                                             : R"(
CHECK: store atomic i64 {{.*}}unordered, align 8
)"));
}

TEST_F(GpuAtomicTest, TestAddAtomicC64) {
  const char* hlo_string = R"(
    HloModule TensorFlowScatterV1

    update_c64 (lhs: c64[], rhs: c64[]) -> c64[] {
      lhs = c64[] parameter(0)
      rhs = c64[] parameter(1)
      ROOT add = c64[] add(lhs, rhs)
    }

    ENTRY main {
      operand = c64[3,3] parameter(0)
      indices = s32[2] parameter(1)
      updates = c64[2,3] parameter(2)
      ROOT scatter = c64[3,3] scatter(operand, indices, updates),
          to_apply=update_c64,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1, unique_indices=false
    }
)";

  ASSERT_OK(CompileAndVerifyIr(hlo_string, IsBuiltWithRocm() ? R"(
CHECK: load atomic i64, ptr %[[ADDR:.*]] syncscope("agent-one-as") monotonic, align 8
CHECK: cmpxchg ptr %[[ADDR]], i64 %[[OLD:.*]], i64 %[[NEW:.*]] syncscope("agent-one-as") monotonic monotonic
)"
                                                             : R"(
CHECK: load atomic i64, ptr %[[ADDR:.*]] unordered, align 8
CHECK: cmpxchg ptr %[[ADDR]], i64 %[[OLD:.*]], i64 %[[NEW:.*]] monotonic monotonic
)"));
}

TEST_F(GpuAtomicTest, TestStoreCasF16) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule ScatterStoreCasF16

    update_f16 (lhs: f16[], rhs: f16[]) -> f16[] {
      lhs = f16[] parameter(0)
      ROOT rhs = f16[] parameter(1)
    }

    ENTRY main {
      operand = f16[4] parameter(0)
      indices = s32[2] parameter(1)
      updates = f16[2] parameter(2)
      ROOT scatter = f16[4] scatter(operand, indices, updates),
          to_apply=update_f16,
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=false
    }
  )hlo";

  ASSERT_OK(CompileAndVerifyIr(kHlo, IsBuiltWithRocm() ? R"(
CHECK: load atomic i32, ptr %[[ADDR:.*]] syncscope("agent-one-as") monotonic, align 4
CHECK: cmpxchg ptr %[[ADDR]], i32 %[[OLD:.*]], i32 %[[NEW:.*]] syncscope("agent-one-as") monotonic monotonic
)"
                                                       : R"(
CHECK: load atomic i32, ptr %[[ADDR:.*]] unordered, align 4
CHECK: cmpxchg ptr %[[ADDR]], i32 %[[OLD:.*]], i32 %[[NEW:.*]] monotonic monotonic
)"));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  Literal operand = LiteralUtil::CreateR1<half>(
      {half(1.0f), half(2.0f), half(3.0f), half(4.0f)});
  Literal indices = LiteralUtil::CreateR1<int32_t>({0, 3});
  Literal updates = LiteralUtil::CreateR1<half>({half(7.0f), half(9.0f)});
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module),
                                               {&operand, &indices, &updates}));
  EXPECT_EQ(result.Get<half>({0}), half(7.0f));
  EXPECT_EQ(result.Get<half>({1}), half(2.0f));
  EXPECT_EQ(result.Get<half>({2}), half(3.0f));
  EXPECT_EQ(result.Get<half>({3}), half(9.0f));
}

TEST_F(GpuAtomicTest, TestStoreCasS4) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule ScatterStoreCasS4

    update_s4 (lhs: s4[], rhs: s4[]) -> s4[] {
      lhs = s4[] parameter(0)
      ROOT rhs = s4[] parameter(1)
    }

    ENTRY main {
      operand = s4[8] parameter(0)
      indices = s32[2] parameter(1)
      updates = s4[2] parameter(2)
      ROOT scatter = s4[8] scatter(operand, indices, updates),
          to_apply=update_s4,
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=false
    }
  )hlo";

  ASSERT_OK(CompileAndVerifyIr(kHlo, IsBuiltWithRocm() ? R"(
CHECK: load atomic i32, ptr %[[ADDR:.*]] syncscope("agent-one-as") monotonic, align 4
CHECK: cmpxchg ptr %[[ADDR]], i32 %[[OLD:.*]], i32 %[[NEW:.*]] syncscope("agent-one-as") monotonic monotonic
)"
                                                       : R"(
CHECK: load atomic i32, ptr %[[ADDR:.*]] unordered, align 4
CHECK: cmpxchg ptr %[[ADDR]], i32 %[[OLD:.*]], i32 %[[NEW:.*]] monotonic monotonic
)"));
}

TEST_F(GpuAtomicTest, TestMinAtomicF32) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule ScatterMinF32

    update_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT min = f32[] minimum(lhs, rhs)
    }

    ENTRY main {
      operand = f32[3] parameter(0)
      indices = s32[4] parameter(1)
      updates = f32[4] parameter(2)
      ROOT scatter = f32[3] scatter(operand, indices, updates),
          to_apply=update_f32,
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=false
    }
  )hlo";

  ASSERT_OK(CompileAndVerifyIr(kHlo, IsBuiltWithRocm() ? R"(
CHECK: load atomic i32, ptr %[[ADDR:.*]] syncscope("agent-one-as") monotonic, align 4
CHECK: cmpxchg ptr %[[ADDR]], i32 %[[OLD:.*]], i32 %[[NEW:.*]] syncscope("agent-one-as") monotonic monotonic
)"
                                                       : R"(
CHECK: load atomic i32, ptr %[[ADDR:.*]] unordered, align 4
CHECK: cmpxchg ptr %[[ADDR]], i32 %[[OLD:.*]], i32 %[[NEW:.*]] monotonic monotonic
)"));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  Literal operand = LiteralUtil::CreateR1<float>({5.0f, -1.0f, 3.0f});
  Literal indices = LiteralUtil::CreateR1<int32_t>({0, 0, 1, 2});
  Literal updates = LiteralUtil::CreateR1<float>({2.0f, -4.0f, -7.5f, 8.0f});
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module),
                                               {&operand, &indices, &updates}));
  EXPECT_EQ(result.Get<float>({0}), -4.0f);
  EXPECT_EQ(result.Get<float>({1}), -7.5f);
  EXPECT_EQ(result.Get<float>({2}), 3.0f);
}

TEST_F(GpuAtomicTest, TestMaxAtomicF32) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule ScatterMaxF32

    update_f32 (lhs: f32[], rhs: f32[]) -> f32[] {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT max = f32[] maximum(lhs, rhs)
    }

    ENTRY main {
      operand = f32[4] parameter(0)
      indices = s32[6] parameter(1)
      updates = f32[6] parameter(2)
      ROOT scatter = f32[4] scatter(operand, indices, updates),
          to_apply=update_f32,
          update_window_dims={},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=false
    }
  )hlo";

  ASSERT_OK(CompileAndVerifyIr(kHlo, IsBuiltWithRocm() ? R"(
CHECK: load atomic float, ptr %[[ADDR:.*]] syncscope("agent-one-as") monotonic, align 4
CHECK-DAG: atomicrmw max ptr %[[ADDR]], i32 %{{.*}} syncscope("agent-one-as") monotonic
CHECK-DAG: atomicrmw umin ptr %[[ADDR]], i32 %{{.*}} syncscope("agent-one-as") monotonic
)"
                                                       : R"(
CHECK: load atomic float, ptr %[[ADDR:.*]] unordered, align 4
CHECK-DAG: atomicrmw max ptr %[[ADDR]], i32 %{{.*}} monotonic
CHECK-DAG: atomicrmw umin ptr %[[ADDR]], i32 %{{.*}} monotonic
)"));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  const float neg_nan = -std::numeric_limits<float>::quiet_NaN();
  Literal operand = LiteralUtil::CreateR1<float>({1.0f, -5.0f, 10.0f, 2.0f});
  Literal indices = LiteralUtil::CreateR1<int32_t>({0, 0, 1, 1, 2, 3});
  Literal updates =
      LiteralUtil::CreateR1<float>({3.5f, 2.0f, -2.0f, -8.0f, 4.0f, neg_nan});
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(module),
                                               {&operand, &indices, &updates}));
  EXPECT_EQ(result.Get<float>({0}), 3.5f);
  EXPECT_EQ(result.Get<float>({1}), -2.0f);
  EXPECT_EQ(result.Get<float>({2}), 10.0f);
  EXPECT_TRUE(std::isnan(result.Get<float>({3})));
  EXPECT_FALSE(std::signbit(result.Get<float>({3})));
}

TEST_F(GpuAtomicTest, TestStoreC128) {
  if (IsBuiltWithRocm() || IsBuiltWithOneAPI() ||
      !device_description().cuda_compute_capability().SupportsAllFeaturesOf(
          stream_executor::CudaComputeCapability::Hopper())) {
    GTEST_SKIP() << "128 bit atomic CAS requires Hopper or newer CUDA GPU.";
  }
  constexpr absl::string_view kHlo = R"hlo(
    HloModule ScatterStoreC128, is_scheduled=true,
        input_output_alias={ {}: (0, {}) }

    update_c128 (lhs: c128[], rhs: c128[]) -> c128[] {
      lhs = c128[] parameter(0)
      ROOT rhs = c128[] parameter(1)
    }

    fused_scatter {
      operand = c128[3] parameter(0)
      indices = s32[2,1] parameter(1)
      updates = c128[2,1] parameter(2)
      ROOT scatter = c128[3] scatter(operand, indices, updates),
          to_apply=update_c128,
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=false
    }

    ENTRY main {
      operand = c128[3] parameter(0)
      indices = s32[2,1] parameter(1)
      updates = c128[2,1] parameter(2)
      ROOT fusion = c128[3] fusion(operand, indices, updates),
          kind=kInput, calls=fused_scatter
    }
  )hlo";

  ASSERT_OK(CompileAndVerifyIr(kHlo, R"(
CHECK: load atomic i128, ptr %[[ADDR:.*]] unordered, align 16
CHECK: cmpxchg ptr %[[ADDR]], i128 %[[OLD:.*]], i128 %[[NEW:.*]] monotonic monotonic
)",
                               /*match_optimized_ir=*/false,
                               /*run_optimization_passes=*/false));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> ptx_module,
                       ParseAndReturnVerifiedModule(kHlo));
  CompileAndOptionallyVerifyPtx(std::move(ptx_module), R"(
CHECK: ld.relaxed.sys.global.b128
CHECK: atom.relaxed.sys.global.cas.b128
)",
                                /*run_optimization_passes=*/false);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> exec_module,
                       ParseAndReturnVerifiedModule(kHlo));
  Literal operand = LiteralUtil::CreateR1<complex128>(
      {complex128(1.0, 2.0), complex128(3.0, 4.0), complex128(5.0, 6.0)});
  Literal indices = LiteralUtil::CreateR2<int32_t>({{0}, {2}});
  Literal updates = LiteralUtil::CreateR2<complex128>(
      {{complex128(10.0, 20.0)}, {complex128(30.0, 40.0)}});
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(exec_module),
                                               {&operand, &indices, &updates},
                                               /*run_hlo_passes=*/false));
  EXPECT_EQ(result.Get<complex128>({0}), complex128(10.0, 20.0));
  EXPECT_EQ(result.Get<complex128>({1}), complex128(3.0, 4.0));
  EXPECT_EQ(result.Get<complex128>({2}), complex128(30.0, 40.0));
}

TEST_F(GpuAtomicTest, TestAddAtomicC128) {
  if (IsBuiltWithRocm() || IsBuiltWithOneAPI() ||
      !device_description().cuda_compute_capability().SupportsAllFeaturesOf(
          stream_executor::CudaComputeCapability::Hopper())) {
    GTEST_SKIP() << "128 bit atomic CAS requires Hopper or newer CUDA GPU.";
  }
  constexpr absl::string_view kHlo = R"hlo(
    HloModule ScatterAddC128, is_scheduled=true,
        input_output_alias={ {}: (0, {}) }

    update_c128 (lhs: c128[], rhs: c128[]) -> c128[] {
      lhs = c128[] parameter(0)
      rhs = c128[] parameter(1)
      ROOT add = c128[] add(lhs, rhs)
    }

    fused_scatter {
      operand = c128[2] parameter(0)
      indices = s32[2,1] parameter(1)
      updates = c128[2,1] parameter(2)
      ROOT scatter = c128[2] scatter(operand, indices, updates),
          to_apply=update_c128,
          update_window_dims={1},
          inserted_window_dims={},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1,
          unique_indices=false
    }

    ENTRY main {
      operand = c128[2] parameter(0)
      indices = s32[2,1] parameter(1)
      updates = c128[2,1] parameter(2)
      ROOT fusion = c128[2] fusion(operand, indices, updates),
          kind=kInput, calls=fused_scatter
    }
  )hlo";

  ASSERT_OK(CompileAndVerifyIr(kHlo, R"(
CHECK: load atomic i128, ptr %[[ADDR:.*]] unordered, align 16
CHECK: cmpxchg ptr %[[ADDR]], i128 %[[OLD:.*]], i128 %[[NEW:.*]] monotonic monotonic
)",
                               /*match_optimized_ir=*/false,
                               /*run_optimization_passes=*/false));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> ptx_module,
                       ParseAndReturnVerifiedModule(kHlo));
  CompileAndOptionallyVerifyPtx(std::move(ptx_module), R"(
CHECK: ld.relaxed.sys.global.b128
CHECK: atom.relaxed.sys.global.cas.b128
)",
                                /*run_optimization_passes=*/false);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> exec_module,
                       ParseAndReturnVerifiedModule(kHlo));
  Literal operand = LiteralUtil::CreateR1<complex128>(
      {complex128(1.0, 2.0), complex128(3.0, 4.0)});
  Literal indices = LiteralUtil::CreateR2<int32_t>({{0}, {0}});
  Literal updates = LiteralUtil::CreateR2<complex128>(
      {{complex128(5.0, 7.0)}, {complex128(11.0, 13.0)}});
  ASSERT_OK_AND_ASSIGN(Literal result, Execute(std::move(exec_module),
                                               {&operand, &indices, &updates},
                                               /*run_hlo_passes=*/false));
  EXPECT_EQ(result.Get<complex128>({0}), complex128(17.0, 22.0));
  EXPECT_EQ(result.Get<complex128>({1}), complex128(3.0, 4.0));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
