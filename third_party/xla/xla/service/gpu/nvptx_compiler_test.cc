/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/gpu/nvptx_compiler.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/backends/gpu/transforms/block_scaling_rewriter.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/alias_info.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/gpu/gpu_latency_hiding_scheduler.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/logical_buffer.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

int64_t CountCopies(const HloComputation& computation) {
  int64_t count = 0;
  for (const auto& instruction : computation.instructions()) {
    if (instruction->opcode() == HloOpcode::kCopy) {
      count++;
    }
  }
  return count;
}

int64_t CountCopies(const HloModule& module) {
  int64_t count = 0;
  for (const auto& computation : module.computations()) {
    count += CountCopies(*computation);
  }
  return count;
}

class NVPTXCompilerTest : public HloPjRtGpuTestBase {
 public:
  absl::StatusOr<std::unique_ptr<BufferAssignment>> AssignBuffers(
      HloModule* module) {
    constexpr uint64_t pointer_size = 4;
    const se::DeviceDescription& gpu_device_info = device_description();
    NVPTXCompiler compiler;
    std::unique_ptr<GpuAliasInfo> alias_info =
        compiler.GetAliasInfo(gpu_device_info);
    ABSL_RETURN_IF_ERROR(ScheduleGpuModule(module, pointer_size, gpu_device_info,
                                      &mlir_context_, alias_info.get())
                        .status());

    auto buffer_size_bytes_function =
        [](const BufferValue& buffer_value) -> int64_t {
      return ShapeSizeBytesFunction(pointer_size)(buffer_value.shape());
    };

    return BufferAssigner::Run(
        module, std::make_unique<SequentialHloOrdering>(module->schedule()),
        buffer_size_bytes_function, alias_info.get(),
        /*color_alignment=*/
        [](LogicalBuffer::Color) { return kXlaAllocatedBufferAlignBytes; },
        BufferAssigner::Options{});
  }

 protected:
  mlir::MLIRContext mlir_context_;
};

class NVPTXCompilerTestTriton : public NVPTXCompilerTest {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = NVPTXCompilerTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_cublas_fallback(false);
    return debug_options;
  }
};

constexpr absl::string_view kScaledDotHlo = R"hlo(
  ENTRY e {
    lhs = f8e4m3fn[256,128] parameter(0)
    rhs = f8e4m3fn[256,128] parameter(1)
    lhs_scale = f8e8m0fnu[8,128] parameter(2)
    rhs_scale = f8e8m0fnu[8,128] parameter(3)
    ROOT dot = bf16[128,128] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
        lhs_contracting_dims={0}, rhs_contracting_dims={0}
  })hlo";

class NVPTXCompilerScaledDotTest : public NVPTXCompilerTest {
 public:
  void SetUp() override {
    NVPTXCompilerTest::SetUp();
    if (!CudnnScaledDotHelper::IsAvailable(device_description())) {
      GTEST_SKIP() << "cuDNN scaled dot requires Blackwell and cuDNN 9.10+.";
    }
  }

  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = NVPTXCompilerTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_scaled_dot_with_triton(false);
    debug_options.set_xla_gpu_autotune_level(0);
    return debug_options;
  }

  absl::StatusOr<std::unique_ptr<HloModule>> CompileScaledDot(
      bool disable_binary_libraries) {
    HloModuleConfig config = GetModuleConfigForTest();
    config.mutable_debug_options()
        .set_xla_gpu_experimental_disable_binary_libraries(
            disable_binary_libraries);
    return GetOptimizedModule(kScaledDotHlo, config);
  }

  // `triangular-solve` constrains its RHS (`dot`) to M-minor `{0,1}` layout.
  absl::StatusOr<std::unique_ptr<HloModule>> CompileMMinorScaledDot(
      absl::string_view scale_shape) {
    std::string hlo = absl::Substitute(R"hlo(
      ENTRY e {
        a = f32[128,128] parameter(0)
        lhs = f4e2m1fn[128,256] parameter(1)
        rhs = f4e2m1fn[128,256] parameter(2)
        lhs_scale = $0 parameter(3)
        rhs_scale = $0 parameter(4)
        dot = f32[128,128] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
            lhs_contracting_dims={1}, rhs_contracting_dims={1}
        ROOT trsm = f32[128,128] triangular-solve(a, dot),
            lower=true, unit_diagonal=true, transpose_a=NO_TRANSPOSE
      })hlo",
                                       scale_shape);
    return GetOptimizedModule(hlo, GetModuleConfigForTest());
  }
};

// ContainsInstrWithOpcode recurses into called computations.
bool HasScaledDot(const HloModule& module) {
  return hlo_query::ContainsInstrWithOpcode(module.entry_computation(),
                                            {HloOpcode::kScaledDot});
}

bool HasDot(const HloModule& module) {
  return hlo_query::ContainsInstrWithOpcode(module.entry_computation(),
                                            {HloOpcode::kDot});
}

bool HasCublasLtMatmul(const HloModule& module) {
  return absl::c_any_of(module.computations(), [](const HloComputation* comp) {
    return absl::c_any_of(comp->instructions(), [](const HloInstruction* hlo) {
      return IsCublasLtMatmul(*hlo);
    });
  });
}

bool HasCudnnFusion(const HloModule& module) {
  return absl::c_any_of(
      module.entry_computation()->instructions(),
      [](const HloInstruction* instr) { return IsCudnnFusion(*instr); });
}

TEST_F(NVPTXCompilerScaledDotTest, ScaledDotIsLoweredToCudnnFusion) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       CompileScaledDot(/*disable_binary_libraries=*/false));
  EXPECT_TRUE(HasCudnnFusion(*module));
  EXPECT_TRUE(HasScaledDot(*module));
  EXPECT_FALSE(absl::c_any_of(module->entry_computation()->instructions(),
                              HloPredicateIsOp<HloOpcode::kScaledDot>));
}

TEST_F(NVPTXCompilerScaledDotTest,
       ScaledDotIsDecomposedWithoutBinaryLibraries) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       CompileScaledDot(/*disable_binary_libraries=*/true));
  EXPECT_FALSE(HasCudnnFusion(*module));
  EXPECT_FALSE(HasScaledDot(*module));
  EXPECT_TRUE(HasDot(*module));
}

TEST_F(NVPTXCompilerScaledDotTest, MMinorMxfp4FallsBackToCublasLt) {
  // NVFP4 accepts M-minor output, confirming `dot` survives pre-layout and gets
  // `{0,1}` layout; CUDNN_FISSION rejects M-minor MXFP4, so CUBLASLT_FISSION
  // lowers it to a GEMM.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> nvfp4_module,
                       CompileMMinorScaledDot("f8e4m3fn[128,16]"));
  EXPECT_TRUE(HasCudnnFusion(*nvfp4_module));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> mxfp4_module,
                       CompileMMinorScaledDot("f8e8m0fnu[128,8]"));
  EXPECT_FALSE(HasCudnnFusion(*mxfp4_module));
  EXPECT_FALSE(HasScaledDot(*mxfp4_module));
  EXPECT_TRUE(HasCublasLtMatmul(*mxfp4_module));
}

TEST_F(NVPTXCompilerScaledDotTest,
       ScaledDotIsDecomposedWhenCudnnFissionExcludedFromAutotuneBackends) {
  HloModuleConfig config = GetModuleConfigForTest();
  auto* backends = config.mutable_debug_options()
                       .mutable_xla_gpu_experimental_autotune_backends();
  backends->Clear();
  backends->Add(autotuner::Backend::CUBLASLT);
  backends->Add(autotuner::Backend::CUBLASLT_FISSION);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       GetOptimizedModule(kScaledDotHlo, config));
  EXPECT_FALSE(HasCudnnFusion(*module));
  EXPECT_FALSE(HasScaledDot(*module));
  EXPECT_TRUE(HasCublasLtMatmul(*module));
}

// Plain CUDNN only tunes kDot fusions (not kScaledDot), so when TRITON is
// excluded by autotune_backends or disabled by deterministic mode, the scaled
// dot must decompose pre-layout.
TEST_F(NVPTXCompilerScaledDotTest,
       ScaledDotIsDecomposedWhenTritonDisabledByAutotuneFlags) {
  auto make_config = [&]() {
    HloModuleConfig config = GetModuleConfigForTest();
    DebugOptions& opts = config.mutable_debug_options();
    opts.set_xla_gpu_experimental_scaled_dot_with_triton(true);
    opts.set_xla_gpu_cudnn_gemm_fusion_level(2);
    opts.clear_xla_gpu_experimental_autotune_backends();
    opts.add_xla_gpu_experimental_autotune_backends(autotuner::Backend::CUDNN);
    return config;
  };

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> excluded_module,
                       GetOptimizedModule(kScaledDotHlo, make_config()));
  EXPECT_FALSE(HasScaledDot(*excluded_module));
  EXPECT_TRUE(HasCudnnFusion(*excluded_module));

  HloModuleConfig deterministic = make_config();
  deterministic.mutable_debug_options()
      .add_xla_gpu_experimental_autotune_backends(autotuner::Backend::TRITON);
  deterministic.mutable_debug_options().set_xla_gpu_deterministic_ops(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> deterministic_module,
                       GetOptimizedModule(kScaledDotHlo, deterministic));
  EXPECT_FALSE(HasScaledDot(*deterministic_module));
  EXPECT_TRUE(HasCudnnFusion(*deterministic_module));
}

// Without GemmFusion, neither Triton nor CUDNN_FISSION can see the scaled dot.
TEST_F(NVPTXCompilerScaledDotTest, ScaledDotIsDecomposedWithoutTritonGemm) {
  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options().set_xla_gpu_enable_triton_gemm(false);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       GetOptimizedModule(kScaledDotHlo, config));
  EXPECT_FALSE(HasCudnnFusion(*module));
  EXPECT_FALSE(HasScaledDot(*module));
  EXPECT_TRUE(HasCublasLtMatmul(*module));
}

// Deviceless cuDNN compilation is off unless explicitly set to AUTO or ALWAYS,
// so the scaled dot must decompose before reaching CUDNN_FISSION.
TEST_F(NVPTXCompilerScaledDotTest,
       ScaledDotIsDecomposedInDevicelessCompilation) {
  for (auto mode : {DebugOptions::CUDNN_DEVICELESS_COMPILATION_UNSET,
                    DebugOptions::CUDNN_DEVICELESS_COMPILATION_DISABLED}) {
    SCOPED_TRACE(DebugOptions::CudnnDevicelessCompilationMode_Name(mode));
    HloModuleConfig config = GetModuleConfigForTest();
    config.mutable_debug_options()
        .set_xla_gpu_cudnn_deviceless_compilation_mode(mode);
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> parsed,
                         ParseAndReturnVerifiedModule(kScaledDotHlo, config));
    Compiler::CompileOptions options;
    options.gpu_topology = GetSingleDeviceGpuTopology("", gpu_target_config());
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<HloModule> module,
        compiler()->RunHloPasses(std::move(parsed),
                                 /*executor=*/nullptr, options));
    EXPECT_FALSE(HasCudnnFusion(*module));
    EXPECT_FALSE(HasScaledDot(*module));
    EXPECT_TRUE(HasDot(*module));
  }
}

TEST_F(NVPTXCompilerScaledDotTest,
       Block16Fp4ScaledDotUsesCudnnWhenTritonEnabled) {
  constexpr absl::string_view kNvfp4Hlo = R"hlo(
    ENTRY e {
      lhs = f4e2m1fn[128,256] parameter(0)
      rhs = f4e2m1fn[128,256] parameter(1)
      lhs_scale = f8e4m3fn[128,16] parameter(2)
      rhs_scale = f8e4m3fn[128,16] parameter(3)
      ROOT dot = bf16[128,128] scaled-dot(lhs, rhs, lhs_scale, rhs_scale),
          lhs_contracting_dims={1}, rhs_contracting_dims={1}
    })hlo";
  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options()
      .set_xla_gpu_experimental_scaled_dot_with_triton(true);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       GetOptimizedModule(kNvfp4Hlo, config));
  EXPECT_TRUE(HasCudnnFusion(*module));
}

TEST_F(NVPTXCompilerScaledDotTest, ScaledDotAutotunesBetweenTritonAndCudnn) {
  HloModuleConfig config = GetModuleConfigForTest();
  DebugOptions& opts = config.mutable_debug_options();
  opts.set_xla_gpu_experimental_scaled_dot_with_triton(true);
  opts.set_xla_gpu_autotune_level(4);
  opts.set_xla_gpu_crash_on_verification_failures(true);
  opts.set_xla_gpu_cublas_fallback(false);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       GetOptimizedModule(kScaledDotHlo, config));
  EXPECT_TRUE(HasScaledDot(*module));
  EXPECT_FALSE(HasCublasLtMatmul(*module));
}

TEST_F(NVPTXCompilerTest, AllReducePerformedInplace) {
  const absl::string_view hlo_string = R"(
HloModule Module, input_output_alias={ {}: (0, {}, may-alias) }

summit {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry {
  param0 = f32[128] parameter(0)
  ROOT allreduce = f32[128] all-reduce(param0),
    replica_groups={}, to_apply=summit
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(auto buffer_assignment, AssignBuffers(module.get()));

  HloInstruction* all_reduce = module->entry_computation()->root_instruction();
  EXPECT_TRUE(buffer_assignment->SharesTopLevelSlice(all_reduce,
                                                     all_reduce->operand(0)));
}

TEST_F(NVPTXCompilerTest, AllReducePerformedInplaceTwoOperands) {
  const absl::string_view hlo_string = R"(
HloModule Module,
  input_output_alias={ {0}: (0, {}, may-alias), {1}: (1, {}, may-alias) }

summit {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}

ENTRY entry {
  param0 = f32[128] parameter(0)
  param1 = f32[128] parameter(1)
  ROOT allreduce = (f32[128], f32[128]) all-reduce(param0, param1),
    replica_groups={}, to_apply=summit
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(auto buffer_assignment, AssignBuffers(module.get()));

  HloInstruction* all_reduce = module->entry_computation()->root_instruction();
  EXPECT_TRUE(buffer_assignment->SharesSliceAtIndex(
      all_reduce, {0}, all_reduce->operand(0), {}));
  EXPECT_TRUE(buffer_assignment->SharesSliceAtIndex(
      all_reduce, {1}, all_reduce->operand(1), {}));
}

TEST_F(NVPTXCompilerTestTriton,
       DotDimensionAreSortedBeforePaddingForCublasEnablingTritonFusion) {
  const absl::string_view hlo_string = R"(
ENTRY e {
 p0 = f16[11,22,33,44] parameter(0)
 p1 = s8[11,22,33,44] parameter(1)
 p1c = f16[11,22,33,44] convert(p1)
 ROOT d = f16[11,22,44,44] dot(p0, p1c),
  lhs_batch_dims={0,1}, lhs_contracting_dims={2},
  rhs_batch_dims={0,1}, rhs_contracting_dims={2}
})";

  se::CudaComputeCapability cc = device_description().cuda_compute_capability();

  if (cc.IsAtLeastAmpere()) {
    MatchOptimizedHlo(hlo_string, R"(
; CHECK: ENTRY
; CHECK: __triton_nested_gemm_fusion
    )");
  } else {
    MatchOptimizedHlo(hlo_string, R"(
; CHECK-NOT: triton
    )");
  }
}

TEST_F(NVPTXCompilerTest, RemovesUnnecessaryCopyInPostSchedulingPipelines) {
  const absl::string_view hlo_text = R"(
HloModule all_gather_overlapping, is_scheduled=true

condition {
  input_tuple = (f32[1,128], f32[2,128], pred[]) parameter(0)
  ROOT cond = pred[] get-tuple-element(input_tuple), index=2
}

body {
  c0 = f32[] constant(0)
  splat_c0 = f32[1,128] broadcast(c0), dimensions={}
  input_tuple = (f32[1,128], f32[2,128], pred[]) parameter(0)
  param_0 = f32[1,128] get-tuple-element(input_tuple), index=0
  add = f32[1,128] add(splat_c0, param_0)
  param_1 = f32[2,128] get-tuple-element(input_tuple), index=1

  c1_s32 = s32[] constant(1)
  c0_s32 = s32[] constant(0)
  dynamic-slice = f32[1,128] dynamic-slice(param_1, c1_s32, c0_s32), dynamic_slice_sizes={1,128}

  // If a schedule was chosen where the all-gather and the dynamic-slice are not
  // intertwined, we can get rid of the copy.
  all-gather-start = (f32[1,128], f32[2,128]) all-gather-start(add), channel_id=1337, replica_groups={{0,1}}, dimensions={0}, use_global_device_ids=true
  all-gather-done = f32[2,128] all-gather-done(all-gather-start)
  copy = f32[2,128] copy(all-gather-done)

  cond = pred[] get-tuple-element(input_tuple), index=2
  ROOT output_tuple = (f32[1,128], f32[2,128], pred[]) tuple(dynamic-slice, copy, cond)
}

ENTRY main {
  param_0 = f32[1,128] parameter(0)
  param_1 = f32[2,128] parameter(1)
  param_2 = pred[] parameter(2)
  copy_param_0 = f32[1,128] copy(param_0)
  copy_param_1 = f32[2,128] copy(param_1)
  tuple = (f32[1,128], f32[2,128], pred[]) tuple(copy_param_0, copy_param_1, param_2)
  while = (f32[1,128], f32[2,128], pred[]) while(tuple), condition=condition, body=body
  get-tuple-element = f32[1,128]{1,0} get-tuple-element((f32[1,128]{1,0}, f32[2,128]{1,0}, pred[]) while), index=0
  get-tuple-element.1 = f32[2,128]{1,0} get-tuple-element((f32[1,128]{1,0}, f32[2,128]{1,0}, pred[]) while), index=1
  get-tuple-element.2 = pred[] get-tuple-element((f32[1,128]{1,0}, f32[2,128]{1,0}, pred[]) while), index=2
  copy.3 = pred[] copy(pred[] get-tuple-element.2)
  ROOT tuple.2 = (f32[1,128]{1,0}, f32[2,128]{1,0}, pred[]) tuple(f32[1,128]{1,0} get-tuple-element, f32[2,128]{1,0} get-tuple-element.1, pred[] copy.3)
}
)";
  auto module = ParseAndReturnVerifiedModule(hlo_text).value();
  EXPECT_EQ(CountCopies(*module), 4);

  const HloInstruction* while_op = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile);
  EXPECT_EQ(while_op->while_body()->root_instruction()->operand(1)->opcode(),
            HloOpcode::kCopy);

  NVPTXCompiler compiler;
  std::unique_ptr<GpuAliasInfo> alias_info =
      compiler.GetAliasInfo(device_description());
  GpuTopology gpu_topology =
      GpuTopology(/*platform_version=*/"", 1, 1, 1, gpu_target_config());
  EXPECT_OK(compiler.RunPostSchedulingPipelines(
      module.get(), 100000, gpu_topology, alias_info.get()));
  EXPECT_EQ(CountCopies(*module), 3);
  while_op = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile);
  // Make sure that the copy of the all-gather has been removed.
  EXPECT_EQ(while_op->while_body()->root_instruction()->operand(1)->opcode(),
            HloOpcode::kAllGather);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
