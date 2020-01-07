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

#include "tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.h"

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
#include "tensorflow/compiler/xla/service/gpu/hlo_algorithm_blacklist.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logger.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#include "tensorflow/stream_executor/gpu/redzone_allocator.h"

namespace xla {
namespace gpu {
namespace {

using absl::optional;
using se::DeviceMemoryBase;
using se::dnn::AlgorithmDesc;
using tensorflow::AutotuneResult;

class ScratchAllocator : public se::ScratchAllocator {
 public:
  ScratchAllocator(int device_ordinal,
                   se::DeviceMemoryAllocator* memory_allocator)
      : device_ordinal_(device_ordinal), memory_allocator_(memory_allocator) {}

  int64 GetMemoryLimitInBytes() override {
    return 1LL << 32;  // 4GB.  TODO(jlebar): Tune this?
  }
  int64 TotalAllocatedBytes() { return total_allocated_bytes_; }

  StatusOr<se::DeviceMemory<uint8>> AllocateBytes(int64 byte_size) override;

  template <typename T>
  StatusOr<se::DeviceMemory<T>> Allocate(int64 num_elements) {
    TF_ASSIGN_OR_RETURN(se::DeviceMemory<uint8> bytes,
                        AllocateBytes(num_elements * sizeof(T)));
    return se::DeviceMemory<T>(bytes);
  }

 private:
  const int device_ordinal_;
  se::DeviceMemoryAllocator* memory_allocator_;
  std::vector<se::OwningDeviceMemory> allocated_buffers_;
  int64 total_allocated_bytes_ = 0;
};

StatusOr<se::DeviceMemory<uint8>> ScratchAllocator::AllocateBytes(
    int64 byte_size) {
  CHECK_GE(byte_size, 0) << "byte_size must be positive.";
  if (byte_size > GetMemoryLimitInBytes()) {
    return se::port::Status(
        se::port::error::RESOURCE_EXHAUSTED,
        absl::StrFormat(
            "Allocating %d bytes exceeds the memory limit of %d bytes.",
            byte_size, GetMemoryLimitInBytes()));
  }

  TF_ASSIGN_OR_RETURN(se::OwningDeviceMemory allocated_buffer,
                      memory_allocator_->Allocate(device_ordinal_, byte_size,
                                                  /*retry_on_failure=*/false));
  total_allocated_bytes_ += byte_size;

  se::DeviceMemoryBase buffer_addr = *allocated_buffer;
  allocated_buffers_.push_back(std::move(allocated_buffer));
  return se::DeviceMemory<uint8>(buffer_addr);
}

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
// Returns a status if an unexpected error has occurred, and the stream
// has been poisoned.
//
// `name` is a user-friendly name for the set of redzones being checked, e.g.
// "input/output" or "scratch".
StatusOr<bool> CheckRedzones(const se::RedzoneAllocator& allocator,
                             se::Stream* stream, absl::string_view name,
                             const HloInstruction* instr,
                             AutotuneResult* result) {
  XLA_SCOPED_LOGGING_TIMER_LEVEL("CudnnConvAlgorithmPicker checking redzones",
                                 2);
  using RedzoneCheckStatus = se::RedzoneAllocator::RedzoneCheckStatus;
  TF_ASSIGN_OR_RETURN(RedzoneCheckStatus redzone_check,
                      allocator.CheckRedzones());
  if (redzone_check.ok()) {
    return true;
  }

  auto* fail = result->mutable_failure();
  fail->set_kind(AutotuneResult::REDZONE_MODIFIED);
  *fail->mutable_msg() = redzone_check.RedzoneFailureMsg();
  fail->set_buffer_address(
      reinterpret_cast<uint64>(redzone_check.user_buffer_address));

  LOG(ERROR) << absl::StreamFormat(
      "Detected cudnn out-of-bounds write in conv %s buffer! This is likely a "
      "cudnn bug. We will skip this algorithm in the future, but your GPU "
      "state may already be corrupted, leading to incorrect results. Within "
      "Google, no action is needed on your part. Outside of Google, please "
      "ensure you're running the latest version of cudnn. If that doesn't fix "
      "the problem, please file a bug with this full error message and we'll "
      "contact nvidia.",
      name);
  LOG(ERROR) << redzone_check.RedzoneFailureMsg();
  LOG(ERROR) << "HloInstruction " << instr->ToString();
  PrintPlatformInfo(stream);
  return false;
}

using ConvCacheKey =
    std::tuple<se::StreamExecutor*,
               /* conv->ToString(HloPrintOptions::Canonical()) */ std::string>;

struct ConvCacheStats {
  int64 cache_hits = 0;
  int64 cache_misses = 0;

  void LogStats() {
    VLOG(2) << "Cache hits: " << cache_hits;
    VLOG(2) << "Cache misses: " << cache_misses;
  }
};

ConvCacheKey AutotuneCacheKeyfromInstruction(
    const HloCustomCallInstruction* conv, se::StreamExecutor* se) {
  auto options = HloPrintOptions::Canonical();
  options.set_print_backend_config(true);
  return std::make_tuple(se, conv->ToString(options));
}

tensorflow::mutex autotune_cache_lock(tensorflow::LINKER_INITIALIZED);
auto& autotune_cache GUARDED_BY(autotune_cache_lock) =
    *new absl::flat_hash_map<ConvCacheKey, AutotuneResult>();
auto& autotune_cache_stats GUARDED_BY(autotune_cache_lock) =
    *new ConvCacheStats();
}  // anonymous namespace

StatusOr<AutotuneResult> GpuConvAlgorithmPicker::PickBestAlgorithm(
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
  ConvCacheKey key = AutotuneCacheKeyfromInstruction(instr, stream_exec_);
  {
    tensorflow::mutex_lock lock(autotune_cache_lock);
    auto it = autotune_cache.find(key);
    if (it != autotune_cache.end()) {
      autotune_cache_stats.cache_hits++;
      return it->second;
    }
    autotune_cache_stats.cache_misses++;
  }

  // Make sure any previous activity on this executor is done. We don't want to
  // interfere with programs that are still running on the GPU.
  if (!stream_exec_->SynchronizeAllActivity()) {
    return InternalError("Failed to synchronize GPU for autotuning.");
  }

  // allocator either points to this->allocator_ or, if that's null, to a
  // se::StreamExecutorMemoryAllocator for stream_exec_.
  se::DeviceMemoryAllocator* allocator;
  optional<se::StreamExecutorMemoryAllocator> se_allocator;
  if (allocator_ != nullptr) {
    allocator = allocator_;
  } else {
    se_allocator.emplace(stream_exec_);
    allocator = &*se_allocator;
  }

  TF_ASSIGN_OR_RETURN(se::Stream* const stream,
                      allocator->GetStream(stream_exec_->device_ordinal()));
  StatusOr<AutotuneResult> result_or(InternalError("Unknown platform."));
  // Check StreamExecutor on which platform it is. ROCm and Cuda implementation
  // have diverged. Specifically, we need to make sure redzone allocator related
  // utilities are not used in ROCm routine
  if (stream_exec_->platform_kind() == se::PlatformKind::kROCm) {
    result_or = PickBestAlgorithmNoCacheRocm(instr, allocator, stream);
  } else if (stream_exec_->platform_kind() == se::PlatformKind::kCuda) {
    result_or = PickBestAlgorithmNoCacheCuda(instr, allocator, stream);
  }

  if (result_or.ok()) {
    tensorflow::mutex_lock lock(autotune_cache_lock);
    CHECK(autotune_cache.insert({key, result_or.ValueOrDie()}).second);
  }
  return result_or;
}

StatusOr<tensorflow::AutotuneResult>
GpuConvAlgorithmPicker::PickBestAlgorithmNoCacheCuda(
    const HloCustomCallInstruction* instr, se::DeviceMemoryAllocator* allocator,
    se::Stream* stream) {
  // Right now Redzone allocator is available in Cuda target only
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "GpuConvAlgorithmPicker::PickBestAlgorithmImpl for ", instr->ToString()));

  const Shape& result_shape = instr->shape().tuple_shapes(0);
  int64 rng_state = 0;

  const auto initialize_buffer = [&stream, &rng_state](
                                     DeviceMemoryBase buffer,
                                     const Shape& buffer_shape) {
    InitializeBuffer(stream, buffer_shape.element_type(), &rng_state, buffer);
  };

  const HloModuleConfig& hlo_module_config = instr->GetModule()->config();

  // Allocate space for the input, filter, and output of the convolution.
  se::RedzoneAllocator input_output_allocator(
      stream, allocator, PtxOptsFromConfig(hlo_module_config));
  std::vector<se::DeviceMemoryBase> operand_buffers;
  for (const auto* operand : instr->operands()) {
    TF_ASSIGN_OR_RETURN(auto buffer,
                        input_output_allocator.AllocateBytes(
                            ShapeUtil::ByteSizeOf(operand->shape())));
    initialize_buffer(buffer, operand->shape());
    operand_buffers.push_back(buffer);
  }
  TF_ASSIGN_OR_RETURN(auto result_buffer,
                      input_output_allocator.AllocateBytes(
                          ShapeUtil::ByteSizeOf(result_shape)));
  initialize_buffer(result_buffer, result_shape);

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      instr->backend_config<CudnnConvBackendConfig>());

  optional<BufferComparator> comparator;
  // Use the first algorithm that's supported as reference. There isn't a
  // particular reason to use it, as any algorithm suffices. It doesn't make
  // this algorithm considered correct, though.
  se::DeviceMemoryBase reference_result_buffer;
  AlgorithmDesc first_algorithm;

  TF_ASSIGN_OR_RETURN(CudnnConvKind kind, GetCudnnConvKind(instr));
  std::vector<AutotuneResult> profile_results;

  const DebugOptions& debug_options =
      instr->GetModule()->config().debug_options();

  const bool crash_on_checking_failure =
      debug_options.xla_gpu_crash_on_verification_failures();

  const auto canonical_hlo =
      std::get<1>(AutotuneCacheKeyfromInstruction(instr, stream_exec_));

  string blas_version;
  if (auto* blas = stream_exec_->AsBlas()) {
    (void)blas->GetVersion(&blas_version);
  }

  absl::Span<const AlgorithmDesc> blacklisted_algos =
      GetBlacklistedConvAlgorithms(GetComputeCapability(stream_exec_),
                                   GetCudnnVersion(stream_exec_), blas_version,
                                   canonical_hlo);

  for (const AlgorithmDesc& alg : GetAlgorithms(kind, stream_exec_)) {
    XLA_SCOPED_LOGGING_TIMER_LEVEL(
        absl::StrCat("CudnnConvAlgorithmPicker::PickBestAlgorithm algo ",
                     AlgorithmToString(alg)),
        2);

    if (absl::c_linear_search(blacklisted_algos, alg)) {
      LOG(INFO) << "Omitted potentially buggy algorithm "
                << AlgorithmToString(alg) << " for conv " << instr->ToString();
      continue;
    }

    se::RedzoneAllocator scratch_allocator(
        stream, allocator, PtxOptsFromConfig(hlo_module_config));
    se::dnn::ProfileResult profile_result;
    VLOG(3) << "Trying algorithm " << AlgorithmToString(alg) << " for "
            << instr->ToString();

    // Use assignment instead of brace-list to make GCC 4.9 happy.
    RunConvOptions options;
    options.profile_result = &profile_result;
    options.algo_override = alg;
    Status launch_status =
        RunGpuConv(instr, absl::MakeSpan(operand_buffers), result_buffer,
                   &scratch_allocator, stream, options);

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
    TF_ASSIGN_OR_RETURN(bool input_output_allocator_redzone_clear,
                        CheckRedzones(input_output_allocator, stream,
                                      "input/output", instr, &result));

    TF_ASSIGN_OR_RETURN(
        bool scratch_allocator_redzone_clear,
        CheckRedzones(scratch_allocator, stream, "scratch", instr, &result));

    if (!input_output_allocator_redzone_clear ||
        !scratch_allocator_redzone_clear) {
      AlgorithmBlacklist proto;
      auto entry = proto.add_entries();
      entry->set_hlo(canonical_hlo);
      *entry->mutable_cc() = GetComputeCapability(stream_exec_);
      *entry->mutable_cudnn_version() = GetCudnnVersion(stream_exec_);
      entry->set_blas_version(blas_version);
      auto algo = entry->add_algos();
      algo->set_id(alg.algo_id());
      algo->set_tensor_ops(alg.tensor_ops_enabled());

      LOG(ERROR)
          << "To blacklist this algorithm for this convolution, "
             "copy-paste the following "
             "proto to the blacklist file pointed by XLA_FLAGS "
             "--xla_gpu_algorithm_blacklist_path="
          << GetDebugOptionsFromFlags().xla_gpu_algorithm_blacklist_path()
          << " : " << proto.ShortDebugString();
      continue;
    }

    if (comparator.has_value()) {
      XLA_SCOPED_LOGGING_TIMER_LEVEL("BufferComparator::CompareEqual", 2);
      StatusOr<bool> compare_result = comparator->CompareEqual(
          stream, reference_result_buffer, result_buffer);
      if (!compare_result.ok()) {
        LOG(ERROR) << "Unable to compare " << AlgorithmToString(first_algorithm)
                   << " against " << AlgorithmToString(alg) << " for "
                   << instr->ToString() << ": " << compare_result.status();
        if (compare_result.status().code() ==
            tensorflow::error::RESOURCE_EXHAUSTED) {
          // Possibly OOM. Propagate the error.
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
        PrintPlatformInfo(stream);
        VLOG(1) << "Full module on failure: \n"
                << instr->GetModule()->ToString();
        auto* fail = result.mutable_failure();
        fail->set_kind(AutotuneResult::WRONG_RESULT);
        fail->set_buffer_address(
            reinterpret_cast<uint64>(result_buffer.opaque()));
        auto* reference_conv = fail->mutable_reference_conv();
        reference_conv->set_algorithm(first_algorithm.algo_id());
        reference_conv->set_tensor_ops_enabled(
            first_algorithm.tensor_ops_enabled());
      }
    } else {
      XLA_SCOPED_LOGGING_TIMER_LEVEL("BufferComparator::Create", 2);
      comparator.emplace(result_shape, hlo_module_config);
      TF_ASSIGN_OR_RETURN(
          reference_result_buffer,
          input_output_allocator.AllocateBytes(result_buffer.size()));
      stream->ThenMemcpy(&reference_result_buffer, result_buffer,
                         result_buffer.size());
      first_algorithm = alg;
    }
  }

  // Log the autotuning result.
  {
    tensorflow::AutotuningLog log;
    {
      ConvInstructionLog instr_log;
      *instr_log.mutable_instruction() = instr->ToProto();
      for (int i = 0; i < instr->operand_count(); i++) {
        *instr_log.add_operand_shapes() = instr->operand(i)->shape().ToProto();
        instr_log.add_operand_addresses(
            reinterpret_cast<uint64>(operand_buffers[i].opaque()));
      }
      instr_log.set_result_address(
          reinterpret_cast<uint64>(result_buffer.opaque()));
      log.mutable_instr()->PackFrom(instr_log);
    }
    for (const auto& profile : profile_results) {
      *log.add_results() = profile;
    }
    *log.mutable_compute_capability() = GetComputeCapability(stream_exec_);
    *log.mutable_cudnn_version() = GetCudnnVersion(stream_exec_);
    log.set_device_pci_bus_id(
        stream_exec_->GetDeviceDescription().pci_bus_id());
    log.set_blas_version(blas_version);
    VLOG(1) << "Autotuning result: " << log.ShortDebugString();
    // If we crash on checking failure, we are in a testing/benchmark mode, thus
    // omitting logging through the logger.
    if (!crash_on_checking_failure) {
      tensorflow::Logger::GetSingleton()->LogProto(log);
    }
  }

  // Crash on miscompares and redzone violations if desired.  Do this after
  // logging the autotuning results, otherwise we won't get any data!
  for (const auto& result : profile_results) {
    if (result.has_failure()) {
      CHECK(!crash_on_checking_failure);
    }
  }

  // For now, we ignore WRONG_RESULT failures because false-positives are
  // possible (e.g. perhaps the reference algorithm is the one that's
  // incorrect!).  But we don't ignore REDZONE_MODIFIED failures because they're
  // quite severe and can be detected with high accuracy.
  auto has_failure = [](const AutotuneResult& r) {
    return r.has_failure() &&
           r.failure().kind() != AutotuneResult::WRONG_RESULT;
  };

  // Choose the fastest convolution that doesn't produce a REDZONE_MODIFIED
  // error.
  //
  // TODO(jlebar): We ought to be able to detect redzone reads by noticing NaNs
  // in the output of the conv and skip those.
  //
  // The successful one should have a smaller key, since we are doing
  // min_element. If they are both unsuccessful, keep the earlier one in
  // the vector by comparing pointers.
  auto result_comparison_key = [&has_failure](const AutotuneResult& r) {
    return std::make_tuple(
        has_failure(r),
        tensorflow::proto_utils::FromDurationProto(r.run_time()));
  };
  const auto& best_result = absl::c_min_element(
      profile_results,
      [&](const AutotuneResult& lhs, const AutotuneResult& rhs) {
        return result_comparison_key(lhs) < result_comparison_key(rhs);
      });

  if (best_result != profile_results.end() && !has_failure(*best_result)) {
    return *best_result;
  }

  return InternalError(
      "All algorithms tried for convolution %s failed.  Falling back to "
      "default algorithm.",
      instr->ToString());
}

StatusOr<tensorflow::AutotuneResult>
GpuConvAlgorithmPicker::PickBestAlgorithmNoCacheRocm(
    const HloCustomCallInstruction* instr, se::DeviceMemoryAllocator* allocator,
    se::Stream* stream) {
  XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
      "GpuConvAlgorithmPicker::PickBestAlgorithmImpl for ", instr->ToString()));

  const auto device_ordinal = stream_exec_->device_ordinal();
  std::vector<se::DeviceMemoryBase> operand_buffers;

  ScratchAllocator input_output_allocator(device_ordinal, allocator);
  const auto initialize_buffer = [stream](DeviceMemoryBase buffer) {
    // Although we don't have evidence this matters, zero out the buffers
    // before autotuning.  It's conceivable that using uninitialized memory as
    // the inputs might affect performance if e.g. the inputs contain
    // denormals, and this is easy enough.
    stream->ThenMemZero(&buffer, buffer.size());
  };

  // Allocate space for the input, filter, and output of the convolution.  We
  // use a ScratchAllocator for this instead of calling allocator_ directly so
  // that our allocations don't leak.
  for (const auto* operand : instr->operands()) {
    TF_ASSIGN_OR_RETURN(auto buffer,
                        input_output_allocator.AllocateBytes(
                            ShapeUtil::ByteSizeOf(operand->shape())));
    initialize_buffer(buffer);
    operand_buffers.push_back(buffer);
  }

  TF_ASSIGN_OR_RETURN(
      auto result_buffer,
      input_output_allocator.AllocateBytes(
          ShapeUtil::ByteSizeOf(instr->shape().tuple_shapes(0))));
  initialize_buffer(result_buffer);

  ScratchAllocator scratch_allocator(device_ordinal, allocator);
  se::dnn::ProfileResult profile_result;
  VLOG(3) << "Auto-tuning for " << instr->ToString();
  RunConvOptions options;
  options.profile_result = &profile_result;

  // ROCm: Set the overriding algorithm to empty to remind cudnn_conv_runner
  // that the AlgorithmConfig in running convolution needs to be empty
  options.algo_override = se::dnn::AlgorithmDesc();

  bool launch_ok =
      RunGpuConv(instr, absl::MakeSpan(operand_buffers), result_buffer,
                 &scratch_allocator, stream, options)
          .ok();

  AutotuneResult best_result;
  if (launch_ok && profile_result.is_valid()) {
    best_result.mutable_conv()->set_algorithm(
        profile_result.algorithm().algo_id());
    best_result.mutable_conv()->set_tensor_ops_enabled(
        profile_result.algorithm().tensor_ops_enabled());
    int64 scratch_bytes_used = scratch_allocator.TotalAllocatedBytes();
    best_result.set_scratch_bytes(scratch_bytes_used);
    *best_result.mutable_run_time() = tensorflow::proto_utils::ToDurationProto(
        absl::Milliseconds(profile_result.elapsed_time_in_ms()));

    return best_result;
  }

  return InternalError(
      "All algorithms tried for convolution %s failed.  Falling back to "
      "default algorithm.",
      instr->ToString());
}

StatusOr<bool> GpuConvAlgorithmPicker::RunOnInstruction(HloInstruction* instr) {
  CHECK(IsCustomCallToDnnConvolution(*instr));

  StatusOr<AutotuneResult> best_algo_or =
      PickBestAlgorithm(Cast<HloCustomCallInstruction>(instr));
  if (!best_algo_or.ok()) {
    LOG(WARNING) << "Failed to determine best cudnn convolution algorithm: "
                 << best_algo_or.status()
                 << "\n\nConvolution performance may be suboptimal.";
    return false;
  }

  auto best_algo = std::move(best_algo_or).ValueOrDie();
  VLOG(2) << "Setting cudnn conv to use algorithm "
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

  VLOG(2) << "Replacing convolution " << instr->ToString() << " with "
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

StatusOr<bool> GpuConvAlgorithmPicker::RunOnComputation(
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

StatusOr<bool> GpuConvAlgorithmPicker::Run(HloModule* module) {
  XLA_SCOPED_LOGGING_TIMER("GpuConvAlgorithmPicker");

  if (module->config().debug_options().xla_gpu_disable_autotune()) {
    VLOG(2) << "Convolution auto-tuning disabled, GpuConvAlgorithmPicker "
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
