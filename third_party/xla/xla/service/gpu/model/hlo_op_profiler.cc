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

#include <array>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

#ifdef GOOGLE_CUDA
#include <algorithm>  // IWYU pragma: keep

#include "xla/backends/profiler/gpu/cupti_buffer_events.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#endif

namespace xla {
namespace gpu {

static constexpr std::array<PrimitiveType, 13> dtypes = {
    S8, S16, S32, S64, U8, U16, U32, U64, F16, F32, F64, C64, C128,
};

static constexpr std::array<HloOpcode, 74> ops = {
    // Unary
    // go/keep-sorted start
    HloOpcode::kAbs,
    HloOpcode::kBitcast,
    HloOpcode::kBitcastConvert,
    HloOpcode::kBroadcast,
    HloOpcode::kCbrt,
    HloOpcode::kCeil,
    HloOpcode::kCholesky,
    HloOpcode::kClz,
    HloOpcode::kCollectivePermuteDone,
    HloOpcode::kConvert,
    HloOpcode::kCopy,
    HloOpcode::kCos,
    HloOpcode::kDomain,
    HloOpcode::kErf,
    HloOpcode::kExp,
    HloOpcode::kExpm1,
    HloOpcode::kFft,
    HloOpcode::kFloor,
    HloOpcode::kGetDimensionSize,
    HloOpcode::kGetTupleElement,
    HloOpcode::kImag,
    HloOpcode::kIsFinite,
    HloOpcode::kLog,
    HloOpcode::kLog1p,
    HloOpcode::kLogistic,
    HloOpcode::kNegate,
    HloOpcode::kNot,
    HloOpcode::kPopulationCount,
    HloOpcode::kReal,
    HloOpcode::kReducePrecision,
    HloOpcode::kReshape,
    HloOpcode::kReverse,
    HloOpcode::kRngBitGenerator,
    HloOpcode::kRoundNearestAfz,
    HloOpcode::kRoundNearestEven,
    HloOpcode::kRsqrt,
    HloOpcode::kSign,
    HloOpcode::kSin,
    HloOpcode::kSlice,
    HloOpcode::kSqrt,
    HloOpcode::kTan,
    HloOpcode::kTanh,
    HloOpcode::kTopK,
    HloOpcode::kTranspose,
    // go/keep-sorted end
    // Binary
    // go/keep-sorted start
    HloOpcode::kAdd,
    HloOpcode::kAddDependency,
    HloOpcode::kAnd,
    HloOpcode::kAtan2,
    HloOpcode::kCompare,
    HloOpcode::kConvolution,
    HloOpcode::kDivide,
    HloOpcode::kDot,
    HloOpcode::kGather,
    HloOpcode::kMaximum,
    HloOpcode::kMinimum,
    HloOpcode::kMultiply,
    HloOpcode::kOr,
    HloOpcode::kOutfeed,
    HloOpcode::kPad,
    HloOpcode::kPower,
    HloOpcode::kRemainder,
    HloOpcode::kSetDimensionSize,
    HloOpcode::kShiftLeft,
    HloOpcode::kShiftRightArithmetic,
    HloOpcode::kShiftRightLogical,
    HloOpcode::kStochasticConvert,
    HloOpcode::kSubtract,
    HloOpcode::kTriangularSolve,
    HloOpcode::kXor,
    // go/keep-sorted end
    // TODO(b/443800190): HloOpcode::kComplex
};

static const absl::flat_hash_set<HloOpcode>& TooFastToMeasureOps() {
  static const absl::NoDestructor<absl::flat_hash_set<HloOpcode>> kOps({
      // go/keep-sorted start
      HloOpcode::kAbs, HloOpcode::kAnd, HloOpcode::kBitcast,
      HloOpcode::kBitcastConvert, HloOpcode::kCeil, HloOpcode::kClz,
      HloOpcode::kCopy, HloOpcode::kFloor, HloOpcode::kImag,
      HloOpcode::kIsFinite, HloOpcode::kMaximum, HloOpcode::kMinimum,
      HloOpcode::kNegate, HloOpcode::kNot, HloOpcode::kOr, HloOpcode::kReal,
      HloOpcode::kSign, HloOpcode::kXor
      // go/keep-sorted end
  });
  return *kOps;
}

static const absl::flat_hash_set<HloOpcode>& UnsupportedOps() {
  static const absl::NoDestructor<absl::flat_hash_set<HloOpcode>> kOps({
      // These Opcodes need custom APIs to create instructions that can be
      // used for profiling. They are not created by HloInstruction::CreateUnary
      // or HloInstruction::CreateBinary functions.

      // TODO(444503555): Add support for these Opcodes by using custom APIs.
      // Unary
      // go/keep-sorted start
      HloOpcode::kBitcastConvert,
      HloOpcode::kBroadcast,
      HloOpcode::kCholesky,
      HloOpcode::kCollectivePermuteDone,
      HloOpcode::kConvert,
      HloOpcode::kDomain,
      HloOpcode::kFft,
      HloOpcode::kGetDimensionSize,
      HloOpcode::kGetTupleElement,
      HloOpcode::kOutfeed,
      HloOpcode::kPad,
      HloOpcode::kPower,
      HloOpcode::kReducePrecision,
      HloOpcode::kRemainder,
      HloOpcode::kReshape,
      HloOpcode::kReverse,
      HloOpcode::kRngBitGenerator,
      HloOpcode::kSetDimensionSize,
      HloOpcode::kShiftLeft,
      HloOpcode::kShiftRightArithmetic,
      HloOpcode::kShiftRightLogical,
      HloOpcode::kSlice,
      HloOpcode::kStochasticConvert,
      HloOpcode::kTopK,
      HloOpcode::kTranspose,
      // go/keep-sorted end
      // Binary
      // go/keep-sorted start
      HloOpcode::kAddDependency,
      HloOpcode::kCompare,
      HloOpcode::kConvolution,
      HloOpcode::kDot,
      HloOpcode::kGather,
      HloOpcode::kOutfeed,
      HloOpcode::kPad,
      HloOpcode::kSetDimensionSize,
      HloOpcode::kTriangularSolve,
      // go/keep-sorted end
  });
  return *kOps;
}

absl::Span<const PrimitiveType> HloOpProfiler::AllSupportedDtypes() {
  return absl::MakeConstSpan(dtypes);
}

absl::Span<const HloOpcode> HloOpProfiler::AllSupportedOps() {
  return absl::MakeConstSpan(ops);
}

const absl::flat_hash_set<HloOpcode>& HloOpProfiler::Unsupported() {
  return UnsupportedOps();
}

const absl::flat_hash_set<HloOpcode>& HloOpProfiler::TooFastToMeasure() {
  return TooFastToMeasureOps();
}

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
    absl::c_sort(kernel_times_ns_);
    auto i = kernel_times_ns_.size() / 2;
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
  // TODO(appujee): This only works when the op takes `Shape` as input; i.e.,
  // fails for kComplex for example.
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

HloOpProfiler::HloOpProfiler(HloRunnerInterface* const absl_nonnull runner,
                             const stream_executor::DeviceDescription* const
                             absl_nonnull device_description)
    : runner_(*ABSL_DIE_IF_NULL(runner)),
      dev_info_(*ABSL_DIE_IF_NULL(device_description)),
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
  // If you get "too fast to measure" errors on faster GPUs, try increasing
  // kMaxOpChainLength.
  constexpr int kMaxOpChainLength = 16 * 1024;

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
