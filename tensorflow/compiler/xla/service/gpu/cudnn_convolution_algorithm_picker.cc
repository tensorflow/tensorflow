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

#include "tensorflow/compiler/xla/service/gpu/cudnn_convolution_algorithm_picker.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/mutex.h"

namespace xla {
namespace gpu {
namespace {

using absl::optional;
using se::DeviceMemoryBase;
using se::dnn::AlgorithmConfig;
using se::dnn::AlgorithmDesc;

class ScratchAllocator : public se::ScratchAllocator {
 public:
  ScratchAllocator(int device_ordinal, DeviceMemoryAllocator* memory_allocator)
      : device_ordinal_(device_ordinal), memory_allocator_(memory_allocator) {}

  int64 GetMemoryLimitInBytes(se::Stream* stream) override {
    return 1LL << 32;  // 4GB.  TODO(jlebar): Tune this?
  }
  int64 TotalAllocatedBytes() { return total_allocated_bytes_; }

  StatusOr<se::DeviceMemory<uint8>> AllocateBytes(se::Stream* stream,
                                                  int64 byte_size) override;

 private:
  const int device_ordinal_;
  DeviceMemoryAllocator* memory_allocator_;
  std::vector<OwningDeviceMemory> allocated_buffers_;
  int64 total_allocated_bytes_ = 0;
};

StatusOr<se::DeviceMemory<uint8>> ScratchAllocator::AllocateBytes(
    se::Stream* stream, int64 byte_size) {
  CHECK_GE(byte_size, 0) << "byte_size must be positive.";
  if (byte_size > GetMemoryLimitInBytes(stream)) {
    return se::port::Status(
        se::port::error::RESOURCE_EXHAUSTED,
        absl::StrFormat(
            "Allocating %d bytes exceeds the memory limit of %d bytes.",
            byte_size, GetMemoryLimitInBytes(stream)));
  }

  TF_ASSIGN_OR_RETURN(OwningDeviceMemory allocated_buffer,
                      memory_allocator_->Allocate(device_ordinal_, byte_size,
                                                  /*retry_on_failure=*/false));
  total_allocated_bytes_ += byte_size;

  se::DeviceMemoryBase buffer_addr = allocated_buffer.AsDeviceMemoryBase();
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

}  // anonymous namespace

// We could have caching here so that we don't redo this work for two identical
// convolutions.  Unfortunately our cache key would have to be a tuple
// containing the protos passed to this function, and we have no utility for
// hashing protos.  We could write our own hash functions, but they'd silently
// break if we ever added a field to one of the protos.  Perhaps we could hack
// using the binary-encoded proto as the hash key, on the assumption that two
// protos being binary-equal is a sufficient, if not necessary, condition for
// proper equality.  But that would still leave us open to having unnecessary
// cache misses and doing extra work.  Overall, caching doesn't seem worth the
// trouble, but we may want to revisit this if we ever find a model where
// caching would speed up compilation a lot.
StatusOr<std::tuple<int64, bool, int64>>
CudnnConvolutionAlgorithmPicker::PickBestAlgorithm(
    HloCustomCallInstruction* instr) {
  // TODO(timshen): for now only check fp16. It can be expanded to other types,
  // with some work on the HLO routines.
  const bool cross_check_enabled =
      instr->shape().tuple_shapes(0).element_type() == xla::F16;

  // Don't run this function concurrently on the same GPU.
  //
  // This is a bit of a hack and doesn't protect us against arbitrary concurrent
  // use of a GPU, but it's sufficient to let us compile two HLO modules
  // concurrently and then run them sequentially.
  tensorflow::mutex_lock lock = LockGpu(stream_exec_);

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

  const auto initialize_buffer = [&stream, cross_check_enabled](
                                     DeviceMemoryBase buffer) {
    if (cross_check_enabled) {
      // Broadcast a constant to the buffer, instead of zeroing the buffer. A
      // non-zero constant is useful for the cross checking, because zero-inputs
      // may not always reveal the bugs.
      CHECK_EQ(0, (uintptr_t)buffer.opaque() % 4);
      size_t left_over_bytes = buffer.size() % 4;
      CHECK_EQ(0, left_over_bytes % 2);

      constexpr float kBroadcastedConstant = 0.1f;
      static const Eigen::half halfs[2] = {Eigen::half(kBroadcastedConstant),
                                           Eigen::half(kBroadcastedConstant)};
      uint32 bits;
      static_assert(sizeof(bits) == sizeof(halfs), "");
      memcpy(&bits, halfs, sizeof(bits));

      size_t aligned_size = buffer.size() / 4 * 4;
      stream.ThenMemset32(&buffer, bits, aligned_size);

      DeviceMemoryBase left_over(
          static_cast<char*>(buffer.opaque()) + aligned_size, left_over_bytes);
      stream.ThenMemcpy(&left_over, halfs, left_over_bytes);
    } else {
      // Although we don't have evidence this matters, zero out the buffers
      // before autotuning.  It's conceivable that using uninitialized memory as
      // the inputs might affect performance if e.g. the inputs contain
      // denormals, and this is easy enough.
      stream.ThenMemZero(&buffer, buffer.size());
    }
  };

  // Allocate space for the input, filter, and output of the convolution.  We
  // use a ScratchAllocator for this instead of calling allocator_ directly so
  // that our allocations don't leak.
  ScratchAllocator input_output_allocator(device_ordinal, allocator);
  std::vector<se::DeviceMemoryBase> operand_buffers;
  for (const auto* operand : instr->operands()) {
    TF_ASSIGN_OR_RETURN(auto buffer,
                        input_output_allocator.AllocateBytes(
                            &stream, ShapeUtil::ByteSizeOf(operand->shape())));
    initialize_buffer(buffer);
    operand_buffers.push_back(buffer);
  }
  TF_ASSIGN_OR_RETURN(
      auto result_buffer,
      input_output_allocator.AllocateBytes(
          &stream, ShapeUtil::ByteSizeOf(instr->shape().tuple_shapes(0))));
  initialize_buffer(result_buffer);

  se::dnn::ProfileResult best_result;
  int64 best_result_bytes_used = 0;
  TF_ASSIGN_OR_RETURN(auto backend_config,
                      instr->backend_config<CudnnConvBackendConfig>());

  optional<F16BufferComparator> comparator;
  // Use the first algorithm that's supported as reference. There isn't a
  // particular reason to use it, as any algorithm sufficies. It doesn't make
  // this algorithm considered correct, though.
  optional<AlgorithmDesc> first_algorithm;
  TF_ASSIGN_OR_RETURN(CudnnConvKind kind, GetCudnnConvKind(instr));
  for (const AlgorithmDesc& alg : GetAlgorithms(kind, stream_exec_)) {
    ScratchAllocator scratch_allocator(device_ordinal, allocator);
    se::dnn::ProfileResult profile_result;
    VLOG(3) << "Trying algorithm " << AlgorithmToString(alg) << " for "
            << instr->ToString();

    backend_config.set_algorithm(alg.algo_id());
    backend_config.set_tensor_ops_enabled(alg.tensor_ops_enabled());
    TF_RETURN_IF_ERROR(instr->set_backend_config(backend_config));
    bool launch_ok = RunCudnnConvolution(instr, absl::MakeSpan(operand_buffers),
                                         result_buffer, &scratch_allocator,
                                         &stream, &profile_result)
                         .ok();

    if (launch_ok && profile_result.is_valid()) {
      const bool crash_on_checking_failure =
          instr->GetModule()
              ->config()
              .debug_options()
              .xla_gpu_crash_on_verification_failures();
      if (comparator.has_value()) {
        StatusOr<bool> result = comparator->CompareEqual(
            se::DeviceMemory<Eigen::half>(result_buffer));
        if (!result.ok()) {
          LOG(ERROR) << "Unable to compare "
                     << AlgorithmToString(*first_algorithm) << " against "
                     << AlgorithmToString(alg) << " for " << instr->ToString()
                     << ": " << result.status();
          CHECK(!crash_on_checking_failure);
        } else if (!result.ValueOrDie()) {
          LOG(ERROR) << "Results mismatch between different convolution "
                        "algorithms. This is likely a bug in convolution, or "
                        "an excessive loss of precision in convolution. "
                     << instr->ToString() << " for "
                     << AlgorithmToString(*first_algorithm) << " vs "
                     << AlgorithmToString(alg);
          CHECK(!crash_on_checking_failure);
        }
      } else if (cross_check_enabled) {
        auto comp = F16BufferComparator::Create(
            se::DeviceMemory<Eigen::half>(result_buffer), compiler_, allocator,
            &stream);
        if (comp.ok()) {
          comparator.emplace(comp.ConsumeValueOrDie());
          first_algorithm.emplace(alg);
        } else {
          LOG(ERROR) << "Fail to initialize buffer comparator: "
                     << comp.status() << ", instruction: " << instr->ToString();
          CHECK(!crash_on_checking_failure);
        }
      }
      int64 scratch_bytes_used = scratch_allocator.TotalAllocatedBytes();
      VLOG(3) << "Run of algorithm " << AlgorithmToString(alg)
              << " succeeded, taking " << profile_result.elapsed_time_in_ms()
              << "ms and using " << NumBytesToString(scratch_bytes_used)
              << " of scratch (Best result: "
              << best_result.elapsed_time_in_ms() << "ms, "
              << NumBytesToString(best_result_bytes_used) << " of scratch)";
      if (profile_result.elapsed_time_in_ms() <
          best_result.elapsed_time_in_ms()) {
        best_result = profile_result;
        best_result_bytes_used = scratch_bytes_used;
      }
    } else {
      VLOG(3) << "Run of algorithm " << AlgorithmToString(alg) << " failed.";
    }
  }
  if (best_result.is_valid()) {
    VLOG(2) << "Best algorithm for " << instr->ToString() << ": "
            << AlgorithmToString(best_result.algorithm()) << ", takes "
            << best_result.elapsed_time_in_ms() << "ms, and uses "
            << best_result_bytes_used << "B of scratch memory.";
    return std::make_tuple(best_result.algorithm().algo_id(),
                           best_result.algorithm().tensor_ops_enabled(),
                           best_result_bytes_used);
  }

  return InternalError(
      "All algorithms tried for convolution %s failed.  Falling back to "
      "default algorithm.",
      instr->ToString());
}

StatusOr<bool> CudnnConvolutionAlgorithmPicker::RunOnInstruction(
    HloInstruction* instr) {
  CHECK(IsCustomCallToDnnConvolution(*instr));

  StatusOr<std::tuple<int64, bool, int64>> alg_scratch_and_tc =
      PickBestAlgorithm(Cast<HloCustomCallInstruction>(instr));

  if (!alg_scratch_and_tc.ok()) {
    LOG(ERROR) << alg_scratch_and_tc.status();
    return false;
  }

  int64 algorithm;
  bool tensor_ops_enabled;
  int64 scratch_bytes;

  std::tie(algorithm, tensor_ops_enabled, scratch_bytes) =
      alg_scratch_and_tc.ConsumeValueOrDie();

  VLOG(1) << "Setting cudnn conv to use algorithm " << algorithm << " and "
          << NumBytesToString(scratch_bytes)
          << " of scratch memory: " << instr->ToString()
          << " tensor_ops_enabled: " << tensor_ops_enabled;

  // Replace instr with a new CustomCall which has the correct algorithm, and
  // whose output shape has the appropriate amount of scratch memory.
  HloComputation* computation = instr->parent();
  Shape new_call_shape =
      ShapeUtil::MakeTupleShape({instr->shape().tuple_shapes(0),
                                 ShapeUtil::MakeShape(U8, {scratch_bytes})});

  TF_ASSIGN_OR_RETURN(CudnnConvBackendConfig backend_config,
                      instr->backend_config<CudnnConvBackendConfig>());
  backend_config.set_algorithm(algorithm);
  backend_config.set_tensor_ops_enabled(tensor_ops_enabled);

  HloInstruction* new_call = computation->AddInstruction(
      instr->CloneWithNewOperands(new_call_shape, instr->operands()));

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

StatusOr<bool> CudnnConvolutionAlgorithmPicker::RunOnComputation(
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

StatusOr<bool> CudnnConvolutionAlgorithmPicker::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
