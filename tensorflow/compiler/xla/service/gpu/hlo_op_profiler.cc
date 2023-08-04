/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/hlo_op_profiler.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_op_profile.pb.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {

/*static*/ std::unique_ptr<HloModule> HloOpProfiler::MakeModuleForMeasurements(
    HloOpcode op, PrimitiveType data_type, int64_t n_elements,
    int chain_length) {
  const Shape shape = ShapeUtil::MakeShape(data_type, {n_elements});
  auto module = std::make_unique<HloModule>("m", HloModuleConfig{});
  HloComputation::Builder entry_builder("b");
  HloComputation::Builder fusion_builder("sb");

  if (HloOpcodeArity(op) == 2) {
    HloInstruction* pf0 = fusion_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "pf0"));
    HloInstruction* pf1 = fusion_builder.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "pf1"));
    HloInstruction* last = pf0;
    for (int i = 0; i < chain_length; ++i) {
      last = fusion_builder.AddInstruction(
          HloInstruction::CreateBinary(shape, op, last, pf1));
    }
    HloComputation* subcomp =
        module->AddEmbeddedComputation(fusion_builder.Build());
    HloInstruction* p0 = entry_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "p0"));
    HloInstruction* p1 = entry_builder.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "p1"));
    entry_builder.AddInstruction(HloInstruction::CreateFusion(
        shape, HloInstruction::FusionKind::kLoop, {p0, p1}, subcomp));
  } else if (HloOpcodeArity(op) == 1) {
    HloInstruction* pf = fusion_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "pf"));
    HloInstruction* last = pf;
    for (int i = 0; i < chain_length; ++i) {
      last = fusion_builder.AddInstruction(
          HloInstruction::CreateUnary(shape, op, last));
    }
    HloComputation* subcomp =
        module->AddEmbeddedComputation(fusion_builder.Build());
    HloInstruction* p0 = entry_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "p0"));
    entry_builder.AddInstruction(HloInstruction::CreateFusion(
        shape, HloInstruction::FusionKind::kLoop, {p0}, subcomp));
  } else {
    LOG(FATAL) << "Unsupported opcode: " << HloOpcodeString(op);
  }
  module->AddEntryComputation(entry_builder.Build());
  VLOG(9) << module->ToString();
  return module;
}

StatusOr<absl::Duration> HloOpProfiler::MeasureOpChainDuration(
    HloOpcode op, PrimitiveType data_type, int64_t input_size,
    int chain_length) {
  std::unique_ptr<HloModule> module =
      MakeModuleForMeasurements(op, data_type, input_size, chain_length);

  std::minstd_rand0 engine;
  // Some operations have dynamic duration that depends on the input values.
  // Measure each operation with small and large inputs and average.
  std::vector<Literal> args_small = MakeFakeArguments(module.get(), &engine,
                                                      /*use_large_range=*/false)
                                        .value();
  std::vector<Literal> args_large = MakeFakeArguments(module.get(), &engine,
                                                      /*use_large_range=*/true)
                                        .value();
  const absl::Time t_compile_start = absl::Now();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Executable> ex,
                      runner_.CreateExecutable(std::move(module),
                                               /*run_hlo_passes=*/false));
  if (absl::Now() - t_compile_start > absl::Seconds(10)) {
    return ResourceExhausted("Too slow compilation");
  }

  // Warmup.
  TF_RETURN_IF_ERROR(
      runner_.ExecuteWithExecutable(ex.get(), args_small).status());

  absl::Duration sum = absl::ZeroDuration();
  constexpr int kIterations = 10;
  for (int i = 0; i < kIterations; ++i) {
    ExecutionProfile profile_small;
    TF_RETURN_IF_ERROR(
        runner_.ExecuteWithExecutable(ex.get(), args_small, &profile_small)
            .status());
    ExecutionProfile profile_large;
    TF_RETURN_IF_ERROR(
        runner_.ExecuteWithExecutable(ex.get(), args_large, &profile_large)
            .status());
    sum += absl::Nanoseconds(
        (profile_small.compute_time_ns() + profile_large.compute_time_ns()) /
        2);
  }
  return sum / kIterations;
}

StatusOr<HloInstructionProfile> HloOpProfiler::MeasureClockCyclesPerOp(
    HloOpcode op, bool is_binary, PrimitiveType data_type, int64_t input_size) {
  VLOG(2) << "Measuring " << HloOpcodeString(op) << " "
          << primitive_util::LowercasePrimitiveTypeName(data_type);

  const absl::Duration overheads =
      MeasureOpChainDuration(HloOpcode::kNegate, data_type, input_size,
                             /*chain_length=*/1)
          .value();
  VLOG(3) << "Overheads: " << overheads;

  absl::Duration duration = absl::ZeroDuration();
  int chain_length = 1;
  // Double the length of the operation chain until it becomes measurable
  // compared to the overheads.
  while (duration < 5 * overheads) {
    TF_ASSIGN_OR_RETURN(duration, MeasureOpChainDuration(
                                      op, data_type, input_size, chain_length));
    VLOG(3) << chain_length << "\t" << duration;
    chain_length *= 2;
    if (chain_length > kMaxOpChainLength) {
      VLOG(2) << "The op is too fast to be measured with this method";
      return Unimplemented("op is too fast");
    }
  }

  TF_ASSIGN_OR_RETURN(
      absl::Duration double_duration,
      MeasureOpChainDuration(op, data_type, input_size, chain_length));
  VLOG(3) << chain_length << "\t" << double_duration;

  // The difference between t_double and t corresponds to half of chain_length.
  const absl::Duration time_per_op =
      (double_duration - duration) * 2.0 / chain_length;

  const float clocks_per_nanosecond =
      dev_info_.clock_rate_ghz * 2;  // 2 for FMA
  const int64_t n_clocks =
      absl::ToInt64Nanoseconds(time_per_op) * clocks_per_nanosecond;
  VLOG(3) << time_per_op << " = " << n_clocks << " clock cycles";
  HloInstructionProfile profile;
  profile.mutable_instruction()->mutable_opcode()->assign(HloOpcodeString(op));
  profile.mutable_instruction()->mutable_shape()->set_element_type(data_type);
  profile.set_clock_cycles(n_clocks);
  return profile;
}

}  // namespace gpu
}  // namespace xla
