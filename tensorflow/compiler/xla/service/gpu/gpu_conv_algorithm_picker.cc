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
#include "google/protobuf/any.pb.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"
#include "tensorflow/compiler/xla/service/gpu/convolution_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_autotuning.pb.h"
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

namespace xla {
namespace gpu {

using absl::optional;
using se::DeviceMemoryBase;
using se::dnn::AlgorithmDesc;
using tensorflow::AutotuneResult;

namespace {
using ConvCacheKey =
    std::tuple<se::StreamExecutor*, std::string, std::string, Shape,
               std::vector<Shape>, std::string, std::string, int64>;

struct ConvCacheStats {
  int64 cache_hits = 0;
  int64 cache_misses = 0;

  void LogStats() {
    VLOG(2) << "Cache hits: " << cache_hits;
    VLOG(2) << "Cache misses: " << cache_misses;
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

string NumBytesToString(int64 bytes) {
  return absl::StrCat(tensorflow::strings::HumanReadableNumBytes(bytes), " (",
                      bytes, "B)");
}

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

  // Make sure any previous activity on this executor is done. We don't want to
  // interfere with programs that are still running on the GPU.
  if (!stream_exec_->SynchronizeAllActivity()) {
    return InternalError("Failed to synchronize GPU for autotuning.");
  }

  // Create a stream for us to do our work on
  se::Stream stream{stream_exec_};
  stream.Init();

  // Allocator either points to this->allocator_ or, if that's null, to a
  // StreamExecutorMemoryAllocator for stream_exec_
  se::DeviceMemoryAllocator* allocator;
  optional<se::StreamExecutorMemoryAllocator> se_allocator;
  if (allocator_ != nullptr) {
    allocator = allocator_;
  } else {
    se_allocator.emplace(stream_exec_);
    allocator = &*se_allocator;
  }

  StatusOr<AutotuneResult> result_or =
      PickBestAlgorithmNoCache(*instr, allocator, &stream);
  if (result_or.ok()) {
    tensorflow::mutex_lock lock(autotune_cache_lock);
    CHECK(autotune_cache.insert({key, result_or.ValueOrDie()}).second);
  }
  return result_or;
}

StatusOr<bool> GpuConvAlgorithmPicker::RunOnInstruction(HloInstruction* instr) {
  CHECK(IsCustomCallToDnnConvolution(*instr));

  StatusOr<AutotuneResult> best_algo_or =
      PickBestAlgorithm(Cast<HloCustomCallInstruction>(instr));
  if (!best_algo_or.ok()) {
    LOG(ERROR) << best_algo_or.status();
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
  backend_config.set_scratch_size(best_algo.scratch_bytes());

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
