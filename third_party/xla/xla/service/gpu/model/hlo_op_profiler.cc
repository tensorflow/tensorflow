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

#include "xla/service/gpu/model/hlo_op_profiler.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/test_utils.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

#ifdef GOOGLE_CUDA
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#endif

namespace xla {
namespace gpu {

#ifdef GOOGLE_CUDA
class CuptiKernelTracer : public HloOpProfiler::KernelTracer,
                          public profiler::CuptiTraceCollector {
 public:
  CuptiKernelTracer()
      : profiler::CuptiTraceCollector({}),
        cupti_tracer_(profiler::CuptiTracer::GetCuptiTracerSingleton()) {
    CHECK(cupti_tracer_->IsAvailable());
    profiler::CuptiTracerOptions options;
    options.cbids_selected.push_back(
        // Not interested in API callbacks, but empty list enables them all.
        CUPTI_DRIVER_TRACE_CBID_cu64GLMapBufferObject);
    options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_KERNEL);
    cupti_tracer_->Enable(options, this).IgnoreError();
  }

  uint64_t getMedianKernelTimeNs() && override {
    cupti_tracer_->Disable();  // Also flushes buffer.
    if (kernel_times_ns_.empty()) {
      LOG(ERROR) << "No kernel events";
      return 0;
    }
    std::sort(kernel_times_ns_.begin(), kernel_times_ns_.end());
    size_t i = kernel_times_ns_.size() / 2;
    // Return median value if number of values is odd.
    if (kernel_times_ns_.size() % 2 != 0) {
      return kernel_times_ns_[i];
    }
    // Return average of the two middle values if the number of values is even.
    return (kernel_times_ns_[i - 1] + kernel_times_ns_[i] + 1) / 2;
  }

 private:
  // CuptiTraceCollector
  void AddEvent(profiler::CuptiTracerEvent&& event) override {
    if (event.type == profiler::CuptiTracerEventType::Kernel) {
      kernel_times_ns_.push_back(event.end_time_ns - event.start_time_ns);
    }
    VLOG(5) << "CuptiTracerEvent: " << event.name << ", "
            << event.end_time_ns - event.start_time_ns << "ns";
  }
  void OnEventsDropped(const std::string& reason,
                       uint32_t num_events) override {
    LOG(WARNING) << "Dropped " << num_events << " events: " << reason;
  }
  void Flush() override {}

  profiler::CuptiTracer* cupti_tracer_;
  std::vector<uint64_t> kernel_times_ns_;
};
#else
class CuptiKernelTracer : public HloOpProfiler::KernelTracer {
 public:
  uint64_t getMedianKernelTimeNs() && {
    LOG(FATAL) << "Not built with --config=cuda";
  }
};
#endif

/*static*/ std::unique_ptr<HloOpProfiler::KernelTracer>
HloOpProfiler::GetKernelTracer() {
  return std::make_unique<CuptiKernelTracer>();
}

/*static*/ std::unique_ptr<HloModule> HloOpProfiler::MakeModuleForMeasurements(
    HloOpcode op, PrimitiveType data_type, int chain_length) {
  constexpr int64_t kInputSize = 1;
  const Shape shape = ShapeUtil::MakeShape(data_type, {kInputSize});
  HloModuleConfig config;
  config.set_debug_options(GetDebugOptionsFromFlags());
  auto module = std::make_unique<HloModule>("module", config);

  HloComputation::Builder entry_builder("entry");
  HloComputation::Builder fusion_builder("fusion");

  HloInstruction* pf = fusion_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "pf"));
  HloInstruction* last = pf;
  for (int i = 0; i < chain_length; ++i) {
    switch (HloOpcodeArity(op).value_or(0)) {
      case 1:
        last = fusion_builder.AddInstruction(
            HloInstruction::CreateUnary(shape, op, last));
        break;
      case 2:
        last = fusion_builder.AddInstruction(
            HloInstruction::CreateBinary(shape, op, last, pf));
        break;
      default:
        LOG(FATAL) << "Unsupported opcode: " << HloOpcodeString(op);
    }
  }
  HloComputation* subcomp =
      module->AddEmbeddedComputation(fusion_builder.Build());
  HloInstruction* p0 = entry_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0"));
  entry_builder.AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kLoop, {p0}, subcomp));
  module->AddEntryComputation(entry_builder.Build());
  VLOG(9) << module->ToString();
  return module;
}

absl::StatusOr<absl::Duration> HloOpProfiler::MeasureOpChainDuration(
    HloOpcode op, PrimitiveType data_type, int chain_length) {
#ifndef GOOGLE_CUDA
  return FailedPrecondition("Not built with --config=cuda");
#endif

  std::unique_ptr<HloModule> module =
      MakeModuleForMeasurements(op, data_type, chain_length);
  HloVerifier verifier(/*layout_sensitive=*/true,
                       /*allow_mixed_precision=*/false);
  TF_RETURN_IF_ERROR(verifier.Run(&*module).status());

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
  TF_ASSIGN_OR_RETURN(std::unique_ptr<OpaqueExecutable> ex,
                      runner_.CreateExecutable(std::move(module),
                                               /*run_hlo_passes=*/false));
  if (absl::Now() - t_compile_start > absl::Seconds(10)) {
    return ResourceExhausted("Too slow compilation");
  }

  // Warmup.
  TF_RETURN_IF_ERROR(
      runner_.ExecuteWithExecutable(ex.get(), args_small).status());

  CuptiKernelTracer cupti_tracer;
  for (int i = 0; i < 10; ++i) {  // Run a few times to reduce noise.
    TF_RETURN_IF_ERROR(
        runner_.ExecuteWithExecutable(ex.get(), args_small).status());
    TF_RETURN_IF_ERROR(
        runner_.ExecuteWithExecutable(ex.get(), args_large).status());
  }

  return absl::Nanoseconds(std::move(cupti_tracer).getMedianKernelTimeNs());
}

HloOpProfiler::HloOpProfiler(HloRunner& runner)
    : runner_(runner),
      dev_info_(runner.backend().stream_executors()[0]->GetDeviceDescription()),
      // Twice the runtime of a copy (chain_length = 0) kernel.
      min_duration_(2 * MeasureOpChainDuration(HloOpcode::kNegate, F32, 0)
                            .value_or(absl::ZeroDuration())) {
  VLOG(3) << "Minimum kernel duration: " << min_duration_;
  CHECK_GT(min_duration_, absl::ZeroDuration())
      << "Failed to measure kernel runtime";
}

absl::StatusOr<HloInstructionProfile> HloOpProfiler::MeasureClockCyclesPerOp(
    HloOpcode op, PrimitiveType data_type) {
  VLOG(2) << "Measuring " << HloOpcodeString(op) << " "
          << primitive_util::LowercasePrimitiveTypeName(data_type);

  // Longer chains are too slow to compile.
  constexpr int kMinOpChainLength = 16;
  constexpr int kMaxOpChainLength = 8192;

  absl::Duration duration = absl::ZeroDuration();
  int chain_length = kMinOpChainLength;
  // Double the length of the operation chain until it becomes measurable
  // compared to the overhead.
  do {
    if (chain_length * 2 > kMaxOpChainLength) {
      return FailedPrecondition("%s is too fast to measure",
                                HloOpcodeString(op));
    }
    TF_ASSIGN_OR_RETURN(duration,
                        MeasureOpChainDuration(op, data_type, chain_length));
    VLOG(3) << chain_length << "\t" << duration;
    chain_length *= 2;
  } while (duration < min_duration_);

  TF_ASSIGN_OR_RETURN(absl::Duration double_duration,
                      MeasureOpChainDuration(op, data_type, chain_length));
  VLOG(3) << chain_length << "\t" << double_duration;

  // The difference between t_double and t corresponds to half of chain_length.
  const absl::Duration time_per_op =
      (double_duration - duration) * 2.0 / chain_length;

  const float clocks_per_nanosecond =
      dev_info_.clock_rate_ghz() * 2;  // 2 for FMA
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
