/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_conv_algorithm_picker.h"

#include "google/protobuf/any.pb.h"
#include "absl/algorithm/container.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_autotuning.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/redzone_allocator.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logger.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/proto/proto_utils.h"

namespace xla {
namespace gpu {
namespace {

using absl::optional;
using se::DeviceMemoryBase;
using se::dnn::AlgorithmDesc;
using tensorflow::AutotuneResult;

std::vector<AlgorithmDesc> GetAlgorithms(CudnnConvKind kind,
                                         se::StreamExecutor* stream_exec) {
  std::vector<AlgorithmDesc> algorithms;
  bool succ = false;
  switch (kind) {
    case CudnnConvKind::kBackwardFilter:
      succ =
          stream_exec->GetConvolveBackwardFilterAlgorithms(true, &algorithms);
      break;
    case CudnnConvKind::kBackwardInput:
      succ = stream_exec->GetConvolveBackwardDataAlgorithms(true, &algorithms);
      break;
    case CudnnConvKind::kForward:
    case CudnnConvKind::kForwardActivation:
      succ = stream_exec->GetConvolveAlgorithms(true, &algorithms);
      break;
  }
  DCHECK(succ);

  return algorithms;
}

string AlgorithmToString(const AlgorithmDesc& algo) {
  if (algo.tensor_ops_enabled()) {
    return absl::StrCat(algo.algo_id(), "+TC");
  }
  return absl::StrCat(algo.algo_id());
}

string NumBytesToString(int64 bytes) {
  return absl::StrCat(tensorflow::strings::HumanReadableNumBytes(bytes), " (",
                      bytes, "B)");
}

// Acquires a process-global lock on the device pointed to by the given
// StreamExecutor.
//
// This is used to prevent other XLA instances from trying to autotune on this
// device while we're using it.
tensorflow::mutex_lock LockGpu(const se::StreamExecutor* stream_exec) {
  static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);
  // se::Platform*s are global singletons guaranteed to live forever.
  static auto* mutexes =
      new std::map<std::pair<const se::Platform*, /*device_ordinal*/ int64>,
                   tensorflow::mutex>();

  tensorflow::mutex_lock global_lock(mu);
  auto it = mutexes
                ->emplace(std::piecewise_construct,
                          std::make_tuple(stream_exec->platform(),
                                          stream_exec->device_ordinal()),
                          std::make_tuple())
                .first;
  return tensorflow::mutex_lock{it->second};
}

tensorflow::CudnnVersion GetCudnnVersion(se::StreamExecutor* stream_executor) {
  tensorflow::CudnnVersion cudnn_version;
  if (auto* dnn = stream_executor->AsDnn()) {
    StatusOr<se::dnn::VersionInfo> version_or = dnn->GetVersion();
    if (version_or.ok()) {
      const auto& version = version_or.ValueOrDie();
      cudnn_version.set_major(version.major_version());
      cudnn_version.set_minor(version.minor_version());
      cudnn_version.set_patch(version.patch());
    }
  }
  return cudnn_version;
}

tensorflow::ComputeCapability GetComputeCapability(
    se::StreamExecutor* stream_executor) {
  tensorflow::ComputeCapability cc;
  int cc_major, cc_minor;
  stream_executor->GetDeviceDescription().cuda_compute_capability(&cc_major,
                                                                  &cc_minor);
  cc.set_major(cc_major);
  cc.set_minor(cc_minor);
  return cc;
}

void PrintPlatformInfo(const se::Stream* stream) {
  auto* se = stream->parent();
  const auto& desc = se->GetDeviceDescription();
  LOG(ERROR) << "Device: " << desc.name();
  LOG(ERROR) << "Platform: " << desc.platform_version();
  LOG(ERROR) << "Driver: " << desc.driver_version();
  LOG(ERROR) << "Runtime: " << desc.runtime_version();

  auto* dnn = se->AsDnn();
  if (dnn) {
    auto dnn_version = dnn->GetVersion();
    if (dnn_version.ok()) {
      auto v = dnn_version.ValueOrDie();
      LOG(ERROR) << "cudnn version: " << v.major_version() << "."
                 << v.minor_version() << "." << v.patch();
    }
  }
}

// Returns true if the redzones in `allocator`'s allocations are unmodified.
//
// If the redzones are modified, logs an error, sets the appropriate failure
// bits on `result`, and returns false.
//
// `name` is a user-friendly name for the set of redzones being checked, e.g.
// "input/output" or "scratch".
bool CheckRedzones(const RedzoneAllocator& allocator, se::Stream* stream,
                   absl::string_view name, const HloInstruction* instr,
                   AutotuneResult* result) {
  XLA_SCOPED_LOGGING_TIMER_LEVEL("CudnnConvAlgorithmPicker checking redzones",
                                 2);

  Status status = allocator.CheckRedzones(stream);
  if (status.ok()) {
    return true;
  }

  auto* fail = result->mutable_failure();
  fail->set_kind(AutotuneResult::REDZONE_MODIFIED);
  *fail->mutable_msg() = status.ToString();

  LOG(ERROR) << absl::StreamFormat(
      "Detected cudnn out-of-bounds write in conv %s buffer! This is likely a "
      "cudnn bug. We will skip this algorithm in the future, but your GPU "
      "state may already be corrupted, leading to incorrect results. Within "
      "Google, no action is needed on your part. Outside of Google, please "
      "ensure you're running the latest version of cudnn. If that doesn't fix "
      "the problem, please file a bug with this full error message and we'll "
      "contact nvidia.",
      name);
  LOG(ERROR) << status.ToString();
  LOG(ERROR) << "HloInstruction " << instr->ToString();
  PrintPlatformInfo(stream);
  return false;
}

using ConvCacheKey =
    std::tuple<se::StreamExecutor*, std::string, std::string, Shape,
               std::vector<Shape>, std::string, std::string, int64>;

struct ConvCacheStats {
  int64 cache_hits = 0;
  int64 cache_misses = 0;

  void LogStats() {
    VLOG(1) << "Cache hits: " << cache_hits;
    VLOG(1) << "Cache misses: " << cache_misses;
  }
};

StatusOr<ConvCacheKey> AutotuneCacheKeyfromInstruction(
    const HloCustomCallInstruction* conv, se::StreamExecutor* se) {
  TF_ASSIGN_OR_RETURN(CudnnConvBackendConfig backend_config,
                      conv->backend_config<CudnnConvBackendConfig>());
  std::vector<Shape> operand_shapes;
  absl::c_transform(conv->operands(), std::back_inserter(operand_shapes),
                    [&](const HloInstruction* op) { return op->shape(); });

  return std::make_tuple(
      se, backend_config.SerializeAsString(), conv->custom_call_target(),
      conv->shape(), std::move(operand_shapes),
      conv->window().SerializeAsString(),
      conv->convolution_dimension_numbers().SerializeAsString(),
      conv->feature_group_count());
}

tensorflow::mutex autotune_cache_lock(tensorflow::LINKER_INITIALIZED);
auto& autotune_cache GUARDED_BY(autotune_cache_lock) =
    *new absl::flat_hash_map<ConvCacheKey, AutotuneResult>();
auto& autotune_cache_stats GUARDED_BY(autotune_cache_lock) =
    *new ConvCacheStats();
}  // anonymous namespace

StatusOr<AutotuneResult> CudnnConvAlgorithmPicker::PickBestAlgorithm(
    const HloCustomCallInstruction* instr) {
  // Don't run this function concurrently on the same GPU.
  //
  // This is a bit of a hack and doesn't protect us against arbitrary concurrent
  // use of a GPU, but it's sufficient to let us compile two HLO modules
  // concurrently and then run them sequentially.
  //
  // Putting the lock in here rather than in PickBestAlgorithmNoCache lets us
  // avoid ever doing duplicate work.  If we have a cache miss, only one thread
  // will run PickBestAlgorithmImpl for a particular device.
  tensorflow::mutex_lock lock = LockGpu(stream_exec_);

  // We cache the autotuning results to avoid doing the duplicate work,
  // which can greatly improve both stability (deterministic numeric results
  // within a process for a given input) and performance (2x speedup on some
  // models).
  TF_ASSIGN_OR_RETURN(ConvCacheKey key,
                      AutotuneCacheKeyfromInstruction(instr, stream_exec_));
  {
    tensorflow::mutex_lock lock(autotune_cache_lock);
    auto it = autotune_cache.find(key);
    if (it != autotune_cache.end()) {
      autotune_cache_stats.cache_hits++;
      return it->second;
    }
    autotune_cache_stats.cache_misses++;
  }

  StatusOr<AutotuneResult> result_or = PickBestAlgorithmNoCache(instr);
  if (result_or.ok()) {
    tensorflow::mutex_lock lock(autotune_cache_lock);
    CHECK(autotune_cache.insert({key, result_or.ValueOrDie()}).second);
  }
  return result_or;
}

StatusOr<AutotuneResult> CudnnConvAlgorithmPicker::PickBestAlgorithmNoCache(
    const HloCustomCallInstruction* instr) {
  XLA_SCOPED_LOGGING_TIMER(
      absl::StrCat("CudnnConvAlgorithmPicker::PickBestAlgorithmImpl for ",
                   instr->ToString()));

  const Shape& result_shape = instr->shape().tuple_shapes(0);

  // Make sure any previous activity on this executor is done. We don't want to
  // interfere with programs that are still running on the GPU.
  if (!stream_exec_->SynchronizeAllActivity()) {
    return InternalError("Failed to synchronize GPU for autotuning.");
  }

  // Create a stream for us to do our work on.
  se::Stream stream{stream_exec_};
  stream.Init();
  const auto device_ordinal = stream_exec_->device_ordinal();

  // allocator either points to this->allocator_ or, if that's null, to a
  // StreamExecutorMemoryAllocator for stream_exec_.
  DeviceMemoryAllocator* allocator;
  optional<StreamExecutorMemoryAllocator> se_allocator;
  if (allocator_ != nullptr) {
    allocator = allocator_;
  } else {
    se_allocator.emplace(stream_exec_->platform(),
                         absl::Span<se::StreamExecutor* const>({stream_exec_}));
    allocator = &*se_allocator;
  }

  const auto initialize_buffer = [&stream,
                                  &result_shape](DeviceMemoryBase buffer) {
    constexpr float kBroadcastedConstant = 0.1f;
    switch (result_shape.element_type()) {
      case xla::F16: {
        // Broadcast a constant to the buffer, instead of zeroing the buffer. A
        // non-zero constant is useful for the cross checking, because
        // zero-inputs may not always reveal the bugs.
        CHECK_EQ(0, (uintptr_t)buffer.opaque() % 4);
        size_t left_over_bytes = buffer.size() % 4;
        CHECK_EQ(0, left_over_bytes % 2);

        static const Eigen::half halfs[2] = {Eigen::half(kBroadcastedConstant),
                                             Eigen::half(kBroadcastedConstant)};
        uint32 bits;
        static_assert(sizeof(bits) == sizeof(halfs), "");
        memcpy(&bits, halfs, sizeof(bits));

        size_t aligned_size = buffer.size() / 4 * 4;
        stream.ThenMemset32(&buffer, bits, aligned_size);

        DeviceMemoryBase left_over(
            static_cast<char*>(buffer.opaque()) + aligned_size,
            left_over_bytes);
        stream.ThenMemcpy(&left_over, halfs, left_over_bytes);
        break;
      }
      case xla::F32: {
        uint32 bits;
        memcpy(&bits, &kBroadcastedConstant, sizeof(bits));
        stream.ThenMemset32(&buffer, bits, buffer.size());
        break;
      }
      // TODO(timshen): populate non-zero data for f64.
      default:
        stream.ThenMemZero(&buffer, buffer.size());
    }
  };

  // Allocate space for the input, filter, and output of the convolution.
  RedzoneAllocator input_output_allocator(device_ordinal, allocator);
  std::vector<se::DeviceMemoryBase> operand_buffers;
  for (const auto* operand : instr->operands()) {
    TF_ASSIGN_OR_RETURN(auto buffer,
                        input_output_allocator.AllocateBytes(
                            &stream, ShapeUtil::ByteSizeOf(operand->shape())));
    initialize_buffer(buffer);
    operand_buffers.push_back(buffer);
  }
  TF_ASSIGN_OR_RETURN(auto result_buffer,
                      input_output_allocator.AllocateBytes(
                          &stream, ShapeUtil::ByteSizeOf(result_shape)));
  initialize_buffer(result_buffer);

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      instr->backend_config<CudnnConvBackendConfig>());

  optional<BufferComparator> comparator;
  // Use the first algorithm that's supported as reference. There isn't a
  // particular reason to use it, as any algorithm sufficies. It doesn't make
  // this algorithm considered correct, though.
  se::DeviceMemoryBase reference_result_buffer;
  AlgorithmDesc first_algorithm;

  TF_ASSIGN_OR_RETURN(CudnnConvKind kind, GetCudnnConvKind(instr));
  std::vector<AutotuneResult> profile_results;

  const DebugOptions& debug_options =
      instr->GetModule()->config().debug_options();

  const bool crash_on_checking_failure =
      debug_options.xla_gpu_crash_on_verification_failures();

  for (const AlgorithmDesc& alg : GetAlgorithms(kind, stream_exec_)) {
    XLA_SCOPED_LOGGING_TIMER_LEVEL(
        absl::StrCat("CudnnConvAlgorithmPicker::PickBestAlgorithm algo ",
                     AlgorithmToString(alg)),
        2);

    RedzoneAllocator scratch_allocator(device_ordinal, allocator);
    se::dnn::ProfileResult profile_result;
    VLOG(3) << "Trying algorithm " << AlgorithmToString(alg) << " for "
            << instr->ToString();

    // Use assignment instead of brace-list to make GCC 4.9 happy.
    RunConvOptions options;
    options.profile_result = &profile_result;
    options.algo_override = alg;
    Status launch_status =
        RunCudnnConv(instr, absl::MakeSpan(operand_buffers), result_buffer,
                     &scratch_allocator, &stream, options);

    if (!launch_status.ok()) {
      continue;
    }

    if (!profile_result.is_valid()) {
      continue;
    }

    profile_results.emplace_back();
    AutotuneResult& result = profile_results.back();
    result.mutable_conv()->set_algorithm(alg.algo_id());
    result.mutable_conv()->set_tensor_ops_enabled(alg.tensor_ops_enabled());

    int64 scratch_bytes_used =
        scratch_allocator.TotalAllocatedBytesExcludingRedzones();
    result.set_scratch_bytes(scratch_bytes_used);
    *result.mutable_run_time() = tensorflow::proto_utils::ToDurationProto(
        absl::Milliseconds(profile_result.elapsed_time_in_ms()));

    // Check for writes to redzones.
    if (!CheckRedzones(input_output_allocator, &stream, "input/output", instr,
                       &result) ||
        !CheckRedzones(scratch_allocator, &stream, "scratch", instr, &result)) {
      continue;
    }

    if (comparator.has_value()) {
      XLA_SCOPED_LOGGING_TIMER_LEVEL("BufferComparator::CompareEqual", 2);
      StatusOr<bool> compare_result = comparator->CompareEqual(
          &stream, allocator, reference_result_buffer, result_buffer);
      if (!compare_result.ok()) {
        LOG(ERROR) << "Unable to compare " << AlgorithmToString(first_algorithm)
                   << " against " << AlgorithmToString(alg) << " for "
                   << instr->ToString() << ": " << compare_result.status();
        if (compare_result.status().code() ==
            tensorflow::error::RESOURCE_EXHAUSTED) {
          // Possibly OOM. Propatate the error.
          return compare_result.status();
        }
        CHECK(!crash_on_checking_failure);
      } else if (!compare_result.ValueOrDie()) {
        LOG(ERROR)
            << "Results mismatch between different convolution algorithms. "
               "This is likely a bug/unexpected loss of precision in cudnn.\n"
            << instr->ToString() << " for "
            << AlgorithmToString(first_algorithm) << " vs "
            << AlgorithmToString(alg);
        PrintPlatformInfo(&stream);
        auto* fail = result.mutable_failure();
        fail->set_kind(AutotuneResult::WRONG_RESULT);
        auto* reference_conv = fail->mutable_reference_conv();
        reference_conv->set_algorithm(first_algorithm.algo_id());
        reference_conv->set_tensor_ops_enabled(
            first_algorithm.tensor_ops_enabled());
      }
    } else {
      XLA_SCOPED_LOGGING_TIMER_LEVEL("BufferComparator::Create", 2);
      auto comp =
          BufferComparator::Create(result_shape, stream.parent(), compiler_);
      if (comp.ok()) {
        comparator.emplace(comp.ConsumeValueOrDie());
        reference_result_buffer = result_buffer;
        TF_ASSIGN_OR_RETURN(result_buffer,
                            input_output_allocator.AllocateBytes(
                                &stream, reference_result_buffer.size()));
        initialize_buffer(result_buffer);
        first_algorithm = alg;
      } else {
        LOG(ERROR) << "Fail to initialize buffer comparator: " << comp.status()
                   << ", instruction: " << instr->ToString();
        CHECK(!crash_on_checking_failure);
      }
    }
  }

  // Log the autotuning result.
  {
    tensorflow::AutotuningLog log;
    {
      ConvInstructionLog instr_log;
      *instr_log.mutable_instruction() = instr->ToProto();
      for (const auto* op : instr->operands()) {
        *instr_log.add_operand_shapes() = op->shape().ToProto();
      }
      log.mutable_instr()->PackFrom(instr_log);
    }
    for (const auto& profile : profile_results) {
      *log.add_results() = profile;
    }
    *log.mutable_compute_capability() = GetComputeCapability(stream_exec_);
    *log.mutable_cudnn_version() = GetCudnnVersion(stream_exec_);
    log.set_device_pci_bus_id(
        stream_exec_->GetDeviceDescription().pci_bus_id());
    VLOG(2) << "Autotuning result:\n" << log.DebugString();
    tensorflow::Logger::Singleton()->LogProto(log);
  }

  // Crash on miscompares and redzone violations if desired.  Do this after
  // logging the autotuning results, otherwise we won't get any data!
  for (const auto& result : profile_results) {
    if (result.has_failure()) {
      CHECK(!crash_on_checking_failure);
    }
  }

  // Choose the fastest convolution that doesn't produce a REDZONE_MODIFIED
  // error.
  //
  // For now, we ignore WRONG_RESULT failures because false-positives are
  // possible (e.g. perhaps the reference algorithm is the one that's
  // incorrect!).  But we don't ignore REDZONE_MODIFIED failures because they're
  // quite severe and can be detected with high accuracy.
  //
  // TODO(jlebar): We ought to be able to detect redzone reads by noticing NaNs
  // in the output of the conv and skip those.
  //
  // The successful one should have a smaller key, since we are doing
  // min_element. If they are both unsuccessful, keep the earlier one in
  // the vector by comparing pointers.
  auto result_comparison_key = [](const AutotuneResult& r) {
    return std::make_tuple(
        r.has_failure() && r.failure().kind() != AutotuneResult::WRONG_RESULT,
        tensorflow::proto_utils::FromDurationProto(r.run_time()));
  };
  const auto& best_result = absl::c_min_element(
      profile_results,
      [&](const AutotuneResult& lhs, const AutotuneResult& rhs) {
        return result_comparison_key(lhs) < result_comparison_key(rhs);
      });

  if (best_result != profile_results.end() && !best_result->has_failure()) {
    return *best_result;
  }

  return InternalError(
      "All algorithms tried for convolution %s failed.  Falling back to "
      "default algorithm.",
      instr->ToString());
}

StatusOr<bool> CudnnConvAlgorithmPicker::RunOnInstruction(
    HloInstruction* instr) {
  CHECK(IsCustomCallToDnnConvolution(*instr));

  StatusOr<AutotuneResult> best_algo_or =
      PickBestAlgorithm(Cast<HloCustomCallInstruction>(instr));
  if (!best_algo_or.ok()) {
    LOG(ERROR) << best_algo_or.status();
    return false;
  }

  auto best_algo = std::move(best_algo_or).ValueOrDie();
  VLOG(1) << "Setting cudnn conv to use algorithm "
          << best_algo.conv().algorithm() << " and "
          << NumBytesToString(best_algo.scratch_bytes())
          << " of scratch memory: " << instr->ToString()
          << " tensor_ops_enabled: " << best_algo.conv().tensor_ops_enabled();

  // Replace instr with a new CustomCall which has the correct algorithm, and
  // whose output shape has the appropriate amount of scratch memory.
  HloComputation* computation = instr->parent();
  Shape new_call_shape = ShapeUtil::MakeTupleShape(
      {instr->shape().tuple_shapes(0),
       ShapeUtil::MakeShape(U8, {best_algo.scratch_bytes()})});

  TF_ASSIGN_OR_RETURN(CudnnConvBackendConfig backend_config,
                      instr->backend_config<CudnnConvBackendConfig>());
  backend_config.set_algorithm(best_algo.conv().algorithm());
  backend_config.set_tensor_ops_enabled(best_algo.conv().tensor_ops_enabled());

  HloInstruction* new_call = computation->AddInstruction(
      instr->CloneWithNewOperands(new_call_shape, instr->operands()));

  VLOG(1) << "Replacing convolution " << instr->ToString() << " with "
          << new_call->ToString();

  TF_RETURN_IF_ERROR(new_call->set_backend_config(backend_config));

  // Repackage new_call so it has the same shape as the original call, namely
  // (conv_result, u8[0]).
  HloInstruction* new_tuple =
      computation->AddInstruction(HloInstruction::CreateTuple(
          {computation->AddInstruction(HloInstruction::CreateGetTupleElement(
               new_call_shape.tuple_shapes(0), new_call, 0)),
           computation->AddInstruction(HloInstruction::CreateConstant(
               LiteralUtil::CreateR1<uint8>({})))}));

  TF_RETURN_IF_ERROR(instr->parent()->ReplaceInstruction(instr, new_tuple));
  return true;
}

StatusOr<bool> CudnnConvAlgorithmPicker::RunOnComputation(
    HloComputation* computation) {
  std::vector<HloInstruction*> convs;
  for (auto* instr : computation->instructions()) {
    if (IsCustomCallToDnnConvolution(*instr)) {
      convs.push_back(instr);
    }
  }

  bool changed = false;
  for (auto* instr : convs) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(instr));
    changed |= result;
  }
  return changed;
}

StatusOr<bool> CudnnConvAlgorithmPicker::Run(HloModule* module) {
  XLA_SCOPED_LOGGING_TIMER("CudnnConvAlgorithmPicker");

  if (module->config().debug_options().xla_gpu_disable_autotune()) {
    VLOG(2) << "Convolution auto-tuning disabled, CudnnConvAlgorithmPicker "
               "returning early.";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }

  {
    tensorflow::mutex_lock lock(autotune_cache_lock);
    autotune_cache_stats.LogStats();
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
