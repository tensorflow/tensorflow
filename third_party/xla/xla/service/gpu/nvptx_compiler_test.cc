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

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/backend.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_hlo_schedule.h"
#include "xla/service/hlo_ordering.h"
#include "xla/service/logical_buffer.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

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

class NVPTXCompilerTest : public HloTestBase {
 public:
  absl::StatusOr<std::unique_ptr<BufferAssignment>> AssignBuffers(
      HloModule* module) {
    constexpr uint64_t pointer_size = 4;
    const se::DeviceDescription& gpu_device_info =
        backend().default_stream_executor()->GetDeviceDescription();
    TF_RETURN_IF_ERROR(
        ScheduleGpuModule(module, pointer_size, gpu_device_info).status());

    auto buffer_size_bytes_function =
        [](const BufferValue& buffer_value) -> int64_t {
      return GetSizeOfShape(buffer_value.shape(), pointer_size);
    };

    return BufferAssigner::Run(
        module, std::make_unique<SequentialHloOrdering>(module->schedule()),
        buffer_size_bytes_function,
        /*color_alignment=*/
        [](LogicalBuffer::Color) { return kXlaAllocatedBufferAlignBytes; });
  }
};

class NVPTXCompilerTestTriton : public NVPTXCompilerTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = HloTestBase::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_cublas_fallback(false);
    return debug_options;
  }
};

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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment, AssignBuffers(module.get()));

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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));

  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment, AssignBuffers(module.get()));

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

  se::CudaComputeCapability cc = backend()
                                     .default_stream_executor()
                                     ->GetDeviceDescription()
                                     .cuda_compute_capability();

  if (cc.IsAtLeastAmpere()) {
    MatchOptimizedHlo(hlo_string, R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: __triton_gemm
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
  TF_EXPECT_OK(compiler.RunPostSchedulingPipelines(
      module.get(), 100000,
      backend().default_stream_executor()->GetDeviceDescription()));
  EXPECT_EQ(CountCopies(*module), 3);
  while_op = hlo_query::GetFirstInstructionWithOpcode(
      *module->entry_computation(), HloOpcode::kWhile);
  // Make sure that the copy of AllGatherDone has been removed.
  EXPECT_EQ(while_op->while_body()->root_instruction()->operand(1)->opcode(),
            HloOpcode::kAllGatherDone);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
