/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/priority_fusion.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusion_deduplication_cache.h"
#include "xla/service/gpu/fusion_process_dump.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/fusion_analysis_cache.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/triton_emitter_constraints.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace gpu {

namespace {

bool IsFusible(const HloInstruction& instr) {
  // Side-effecting operations are not fusible.
  if (!instr.IsFusible()) {
    return false;
  }

  // Element-wise operations are always fusible.
  if (instr.IsElementwise()) {
    return true;
  }

  // Other non-elementwise ops also supported by elemental fusion.
  switch (instr.opcode()) {
    case HloOpcode::kFusion:
      return IsGenericTritonFusion(instr) ||
             instr.fusion_kind() != HloInstruction::FusionKind::kCustom;
    case HloOpcode::kCopy:
    case HloOpcode::kIota:
    case HloOpcode::kConstant:
    case HloOpcode::kReduce:
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConcatenate:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kPad:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kScatter:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      return true;
    default:
      return false;
  }
}

// Returns a GpuBackendConfig proto for a Triton fusion with the given
// BlockLevelParameters.
GpuBackendConfig GetTritonGpuBackendConfig(
    const BlockLevelParameters& block_level_parameters) {
  GpuBackendConfig gpu_backend_config;
  gpu_backend_config.mutable_fusion_backend_config()->set_kind(
      std::string(kTritonFusionKind));
  *gpu_backend_config.mutable_fusion_backend_config()
       ->mutable_block_level_fusion_config() =
      block_level_parameters.ToBlockLevelFusionConfig();
  return gpu_backend_config;
}

// An implementation of FusionQueue that determines whether to fuse instructions
// according to a cost model, and chooses the next fusion candidate according to
// dynamically updated priorities. The elements in the queue are producer nodes
// that could be fused, and the priority of a producer is the benefit in
// performance when fusing it to all of its fusible users. We greedily pick the
// max-benefit producer to fuse, and update the estimated benefits of the fused
// nodes and their operands.
class PriorityFusionQueue {
  using Priority = absl::Duration;
  using CanFuseCallback = std::function<FusionDecision(
      HloInstruction* /*producer*/, int64_t /*consumer operand_index*/)>;

 public:
  PriorityFusionQueue(HloComputation* computation,
                      const GpuHloCostAnalysis::Options& cost_analysis_options,
                      const se::DeviceDescription* device_info,
                      FusionProcessDumpProto* fusion_process_dump,
                      tsl::thread::ThreadPool* thread_pool,
                      mlir::MLIRContext* mlir_context,
                      HloFusionAnalysisCache& fusion_analysis_cache,
                      FusionDeduplicationCache& fusion_deduplication_cache,
                      bool triton_heroless_fusion_enabled)
      : computation_(computation),
        device_info_(device_info),
        cost_analysis_(cost_analysis_options, *device_info),
        gpu_indexing_performance_model_(device_info, &fusion_analysis_cache,
                                        cost_analysis_options.shape_size,
                                        mlir_context),
        fusion_process_dump_(fusion_process_dump),
        thread_pool_(thread_pool),
        mlir_context_(mlir_context),
        fusion_analysis_cache_(fusion_analysis_cache),
        fusion_deduplication_cache_(fusion_deduplication_cache),
        fusion_info_cache_(*device_info_),
        triton_heroless_fusion_enabled_(triton_heroless_fusion_enabled) {
    VLOG(2) << "Running full HLO cost analysis for " << computation_->name();
    TF_CHECK_OK(computation_->Accept(&cost_analysis_));

    dump_fusion_visualization_ = computation->parent()
                                     ->config()
                                     .debug_options()
                                     .xla_dump_fusion_visualization();

    // Initializes the priority queue.
    std::vector<HloInstruction*> instructions;
    for (auto* instruction : computation->MakeInstructionPostOrder()) {
      TF_CHECK_OK(UpdatePerformanceModelCache(instruction));
      if (HloPredicateIsOp<HloOpcode::kParameter>(instruction) ||
          instruction->user_count() == 0 || !instruction->IsFusible() ||
          HloPredicateIsOp<HloOpcode::kTuple, HloOpcode::kGetTupleElement>(
              instruction)) {
        continue;
      }
      instructions.push_back(instruction);
    }
    ComputeAndSetPriorities(instructions);
  }

  void ComputeAndSetPriorities(
      const std::vector<HloInstruction*>& instructions) {
    std::vector<Priority> priorities = ComputePriorities(instructions);

    for (auto [instruction, priority] : llvm::zip(instructions, priorities)) {
      auto key = std::make_pair(priority, instruction->unique_id());

      // Remove instruction with the old priority from the queue.
      auto reverse_it = reverse_map_.find(instruction);
      if (reverse_it != reverse_map_.end()) {
        const PriorityQueue::iterator& queue_it = reverse_it->second;
        // Priority didn't change. Nothing to do.
        if (key == queue_it->first) {
          continue;
        }
        producer_priority_queue_.erase(queue_it);
        reverse_map_.erase(reverse_it);
      }

      // If the priority is negative, it's not helpful to perform fusion on this
      // instruction.
      if (priority < absl::ZeroDuration()) {
        continue;
      }

      auto emplace_result = producer_priority_queue_.emplace(key, instruction);
      reverse_map_.emplace(instruction, emplace_result.first);
    }
  }

  std::vector<Priority> ComputePriorities(
      const std::vector<HloInstruction*>& instructions) {
    auto schedule_or_run = [this](std::function<void()> fn) {
      if (thread_pool_) {
        thread_pool_->Schedule(std::move(fn));
      } else {
        fn();
      }
    };
    absl::BlockingCounter counter(instructions.size());
    std::vector<Priority> priorities(instructions.size());

    for (size_t i = 0; i < instructions.size(); ++i) {
      schedule_or_run([&, i] {
        priorities[i] = CalculateProducerPriority(instructions[i]);
        counter.DecrementCount();
      });
    }
    counter.Wait();
    return priorities;
  }

  // Gets the next pair of (producer, consumers) from the queue for fusion.
  // Returns true if there is the next producer to fuse, otherwise false. Stores
  // the producer and consumers in `current_producer_` and `current_consumers_`.
  bool DequeueNextProducer() {
    current_producer_ = nullptr;
    current_consumers_.clear();

    while (!producer_priority_queue_.empty() && current_consumers_.empty()) {
      auto next_it = std::prev(producer_priority_queue_.end());

      current_producer_ = next_it->second;
      producer_priority_queue_.erase(next_it);
      reverse_map_.erase(current_producer_);

      current_consumers_ = current_producer_->users();

      if (HloPredicateIsOp<HloOpcode::kBitcast>(current_producer_)) {
        // We don't check if bitcasts can be fused with all consumers, so we
        // have to do it here.
        llvm::erase_if(current_consumers_, [&](HloInstruction* consumer) {
          return !CanFuseCached(current_producer_, consumer);
        });
      }
    }

    return !current_consumers_.empty();
  }

  absl::Status UpdatePerformanceModelCache(HloInstruction* producer) {
    if (!IsFusible(*producer)) {
      return absl::OkStatus();
    }

    if (gpu_performance_model_cache_.Get(*producer)) {
      return absl::OkStatus();
    }

    EstimateRunTimeData runtime_data;
    if (IsGenericTritonFusion(*producer)) {
      TF_ASSIGN_OR_RETURN(
          runtime_data,
          gpu_indexing_performance_model_.EstimateRunTimeForTriton(producer));
    } else {
      auto config = GpuPerformanceModelOptions::Default(
          &fusion_analysis_cache_, &gpu_performance_model_cache_);
      runtime_data = GpuPerformanceModel::EstimateRunTimeForInstruction(
          producer, *device_info_, &cost_analysis_, config);
    }

    gpu_performance_model_cache_.Set(*producer, runtime_data);

    return absl::OkStatus();
  }

  // Update priorities of all affected ops.
  absl::Status UpdatePriorities() {
    // Revisit costs of all updated ops. It's important to update cost analysis
    // before recalculating priorities.
    for (auto instruction : to_update_priority_) {
      TF_RETURN_IF_ERROR(cost_analysis_.RevisitInstruction(instruction));
    }
    for (auto producer : to_update_priority_) {
      TF_RETURN_IF_ERROR(UpdatePerformanceModelCache(producer));
    }

    ComputeAndSetPriorities(std::vector<HloInstruction*>{
        to_update_priority_.begin(), to_update_priority_.end()});

    to_update_priority_.clear();
    operands_to_new_consumers_.clear();
    operands_to_removed_consumers_runtimes_.clear();
    return absl::OkStatus();
  }

  // Prepares producer and consumer instruction to be fused. Invalidates caches
  // and writes logs.
  void PreFusion(HloInstruction* producer, HloInstruction* consumer) {
    if (dump_fusion_visualization_) {
      RegisterFusionState(
          *computation_,
          absl::StrCat("About to fuse |", producer->name(), "| into |",
                       consumer->name(), "| inside PriorityFusion"),
          *consumer, producer);
    }
  }

  // Invalidates all cached value related to this instruction. Called before the
  // instruction is fused. The instruction can be either producer or consumer.
  void InvalidateCaches(HloInstruction* instruction) {
    can_fuse_cache_.erase(instruction);
    for (const HloInstruction* operand : instruction->operands()) {
      auto it = can_fuse_cache_.find(operand);
      if (it != can_fuse_cache_.end()) {
        it->second.erase(instruction);
      }
    }

    block_level_parameters_cache_.erase(instruction);
    for (const HloInstruction* operand : instruction->operands()) {
      auto it = block_level_parameters_cache_.find(operand);
      if (it != block_level_parameters_cache_.end()) {
        it->second.erase(instruction);
      }
    }

    gpu_performance_model_cache_.Invalidate(*instruction);
    fusion_analysis_cache_.Invalidate(*instruction);
    fusion_info_cache_.Invalidate(instruction);
  }

  void UpdateRuntimes(
      GpuPerformanceModel::RunTimes& runtimes, const HloInstruction* consumer,
      const absl::flat_hash_map<const HloInstruction*, absl::Duration>&
          original_consumers) {
    auto it = original_consumers.find(consumer);
    if (it != original_consumers.end()) {
      runtimes.time_fused += it->second;
      auto consumer_cache_result = gpu_performance_model_cache_.Get(*consumer);
      CHECK(consumer_cache_result.has_value());
      runtimes.time_unfused += (*consumer_cache_result).exec_time;
    }
  }

  // Prepare for incremental updates
  void ComputeRuntimesOfRemovedConsumers() {
    for (const auto& pair : operands_to_new_consumers_) {
      auto operand = pair.first;
      // Checks if this producer's priority was calculated before. If so, we can
      // do incremental update here.
      if (!reverse_map_.contains(operand)) {
        continue;
      }
      // Get all of this producer's original consumers. Bitcast/constant have
      // priority calculated but they don't have cache entries.
      if (!gpu_performance_model_cache_.ContainsConsumers(*operand)) {
        continue;
      }
      const auto& original_consumers =
          gpu_performance_model_cache_.GetAllConsumers(*operand);
      GpuPerformanceModel::RunTimes runtimes;
      for (auto consumer : current_consumers()) {
        UpdateRuntimes(runtimes, consumer, original_consumers);
      }
      UpdateRuntimes(runtimes, current_producer(), original_consumers);
      auto operand_cache_result = gpu_performance_model_cache_.Get(*operand);
      runtimes.time_unfused += (*operand_cache_result).exec_time +
                               GpuPerformanceModel::kKernelLaunchOverhead;
      operands_to_removed_consumers_runtimes_.emplace(operand, runtimes);
    }
  }

  // Updates data for the new fusion instruction and its users and operands.
  void OnFusingInstruction(HloInstruction* fusion,
                           HloInstruction* original_producer,
                           HloInstruction* original_consumer) {
    if (fusion_process_dump_) {
      auto* fusion_step =
          fusion_process_dump_->add_fusion_steps()->mutable_fusion();

      // Explicit std::string is needed for OSS proto implementation.
      fusion_step->set_fusion_name(std::string(fusion->name()));
      fusion_step->set_producer_name(std::string(original_producer->name()));
      fusion_step->set_consumer_name(std::string(original_consumer->name()));
    }

    if (dump_fusion_visualization_) {
      RegisterFusionState(
          *computation_,
          absl::StrCat("Fused |", original_producer->name(), "| into |",
                       fusion->name(), "| inside PriorityFusion"),
          *fusion);
    }

    // The original consumer was replaced with the fusion, but it's pointer can
    // still be referenced somewhere, for example, in to_update_priority_.
    // Priority recomputation is called before DCE. Remove all references to
    // the original consumer here.
    if (fusion != original_consumer) {
      RemoveInstruction(original_consumer);
    }

    // Collect the instructions whose priorities need to be updated.
    for (HloInstruction* operand : fusion->operands()) {
      if (operand == original_producer ||
          HloPredicateIsOp<HloOpcode::kConstant, HloOpcode::kGetTupleElement>(
              operand)) {
        continue;
      }
      // Need to consider only instructions that are fusible, e.g., rng with
      // greater than one user is not fusible.
      if (!operand->IsFusible()) {
        continue;
      }

      to_update_priority_.insert(operand);
      // update the consumers of this operand that we care about,
      // so we can do incremental update of the operand
      operands_to_new_consumers_[operand].push_back(fusion);
    }
    to_update_priority_.insert(fusion);
  }

  // Removes data for the instruction.
  void RemoveInstruction(HloInstruction* instruction) {
    to_update_priority_.erase(instruction);
    fusion_analysis_cache_.Invalidate(*instruction);

    auto reverse_it = reverse_map_.find(instruction);
    if (reverse_it == reverse_map_.end()) {
      return;
    }
    producer_priority_queue_.erase(reverse_it->second);
    reverse_map_.erase(reverse_it);
  }

  // Returns a map from consumer to BlockLevelParameters. This is used to
  // determine if a producer-consumer fusion is a Triton fusion.
  absl::flat_hash_map<const HloInstruction*, BlockLevelParameters>
  GetBlockLevelParametersMap(const HloInstruction* producer) {
    auto it = block_level_parameters_cache_.find(producer);
    if (it == block_level_parameters_cache_.end()) {
      return {};
    }
    return it->second;
  }

  HloInstruction* current_producer() { return current_producer_; }

  const std::vector<HloInstruction*>& current_consumers() {
    return current_consumers_;
  }

 private:
  // Returns the priority of the producer based on its current operands and
  // users.
  Priority CalculateProducerPriority(HloInstruction* producer) {
    // Bitcasts should always be fused first, since they are no-ops.
    if (HloPredicateIsOp<HloOpcode::kBitcast>(producer)) {
      return absl::InfiniteDuration();
    }
    // We always fuse constants, but the cost model doesn't handle them very
    // well: fusing constants changes costs significantly. Also, there's no
    // point recomputing priorities. Therefore, we fuse all of them at the end.
    if (HloPredicateIsOp<HloOpcode::kConstant>(producer)) {
      return -absl::InfiniteDuration();
    }

    // Don't fuse if we can't fuse in all users.
    if (auto fusion_decision = CanFuseWithAllNonBitcastUsers(producer);
        !fusion_decision) {
      if (fusion_process_dump_) {
        absl::MutexLock lock(&fusion_process_dump_mutex_);
        auto* step = fusion_process_dump_->add_fusion_steps()
                         ->mutable_producer_ineligible();
        step->set_producer_name(std::string(producer->name()));
        step->set_reason(fusion_decision.Explain());
      }
      return -absl::InfiniteDuration();
    }

    auto removed_consumers_runtime_it =
        operands_to_removed_consumers_runtimes_.find(producer);
    bool is_incremental_update = removed_consumers_runtime_it !=
                                 operands_to_removed_consumers_runtimes_.end();
    absl::Span<HloInstruction* const> fused_consumers =
        is_incremental_update
            ? operands_to_new_consumers_.find(producer)->second
            : absl::MakeConstSpan(producer->users());
    // Note that `gpu_performance_model_cache_` may contain a runtime estimate
    // from the Triton cost model.
    GpuPerformanceModel::RunTimes run_times =
        GpuPerformanceModel::EstimateRunTimes(
            producer, *device_info_, &cost_analysis_,
            GpuPerformanceModelOptions::Default(&fusion_analysis_cache_,
                                                &gpu_performance_model_cache_),
            fused_consumers);
    Priority current_priority;
    if (is_incremental_update) {
      // subtract the runtimes of removed consumers
      const GpuPerformanceModel::RunTimes& removed_consumers_runtime =
          removed_consumers_runtime_it->second;
      run_times.time_unfused -= removed_consumers_runtime.time_unfused;
      run_times.time_fused -= removed_consumers_runtime.time_fused;
      // get the original priority
      const PriorityQueue::iterator& queue_it =
          FindOrDie(reverse_map_, producer);
      current_priority = queue_it->first.first;
    }

    if (fusion_process_dump_) {
      absl::MutexLock lock(&fusion_process_dump_mutex_);
      auto* step =
          fusion_process_dump_->add_fusion_steps()->mutable_update_priority();
      step->set_producer_name(std::string(producer->name()));
      for (auto* consumer : producer->users()) {
        step->add_consumer_names(std::string(consumer->name()));
      }
      step->set_us_fused(absl::ToDoubleMicroseconds(run_times.time_fused));
      step->set_us_unfused(absl::ToDoubleMicroseconds(run_times.time_unfused));
    }
    return current_priority + run_times.time_unfused - run_times.time_fused;
  }

  FusionDecision IsTritonSupported(const HloInstruction& instruction) {
    if (IsGenericTritonFusion(instruction)) {
      return FusionDecision::Allow();
    }

    if (instruction.opcode() != HloOpcode::kFusion) {
      return IsTritonSupportedInstruction(
          instruction, device_info_->gpu_compute_capability());
    }

    for (const HloInstruction* instruction :
         instruction.fused_instructions_computation()->instructions()) {
      if (auto codegen_decision = IsTritonSupportedInstruction(
              *instruction, device_info_->gpu_compute_capability());
          !codegen_decision) {
        return codegen_decision;
      }
    }

    return FusionDecision::Allow();
  }

  TiledRunTimeDataOrError GetTiledRunTimeDataCached(
      const HloInstruction* producer, const HloInstruction* consumer) {
    FusionDeduplicationCache::FusionId fusion_id = [&]() {
      absl::MutexLock lock(&fusion_deduplication_cache_mutex_);
      return fusion_deduplication_cache_.GetFusionId(producer, consumer);
    }();

    {
      absl::MutexLock lock(&tiled_run_time_data_cache_mutex_);

      auto it = tiled_run_time_data_cache_.find(fusion_id);
      if (it != tiled_run_time_data_cache_.end()) {
        return it->second;
      }
    }

    auto fusion = HloFusionAdaptor::ForProducerConsumer(producer, consumer);

    absl::StatusOr<TiledRunTimeDataOrError> result_or_status =
        gpu_indexing_performance_model_.TryFindBestTilingForFusion(*fusion);

    // Convert absl::Status into FusionDecision. We don't distinguish between
    // status and FusionDecision here, because both indicate that tile analysis
    // failed and we shouldn't fuse.
    TiledRunTimeDataOrError tiled_run_time_data_or_error =
        [&]() -> TiledRunTimeDataOrError {
      if (result_or_status.ok()) {
        return *result_or_status;
      } else {
        return FusionDecision::Forbid(
            absl::StrCat("TiledRunTimeDataOrError return status: ",
                         result_or_status.status().message()));
      }
    }();

    if (const auto* fusion_decision =
            std::get_if<FusionDecision>(&tiled_run_time_data_or_error)) {
      tiled_run_time_data_or_error = FusionDecision::Forbid(
          absl::StrCat("Fusion can not be tiled with SymbolicTileAnalysis: ",
                       fusion_decision->Explain()));
    }

    absl::MutexLock lock(&tiled_run_time_data_cache_mutex_);
    tiled_run_time_data_cache_.emplace(fusion_id, tiled_run_time_data_or_error);
    return tiled_run_time_data_or_error;
  }

  FusionDecision CanFuseTriton(HloInstruction* producer,
                               HloInstruction* consumer) {
    if (!IsGenericTritonFusion(*producer) &&
        !IsGenericTritonFusion(*consumer) && !triton_heroless_fusion_enabled_) {
      return FusionDecision::Forbid("triton heroless fusion is not enabled");
    }

    if (auto fusion_decision = IsTritonSupported(*producer); !fusion_decision) {
      return fusion_decision;
    }

    if (auto fusion_decision = IsTritonSupported(*consumer); !fusion_decision) {
      return fusion_decision;
    }

    TiledRunTimeDataOrError tiled_run_time_data_or_error =
        GetTiledRunTimeDataCached(producer, consumer);

    if (const auto* fusion_decision =
            std::get_if<FusionDecision>(&tiled_run_time_data_or_error)) {
      return *fusion_decision;
    }

    TiledRunTimeData tiled_run_time_data =
        std::get<TiledRunTimeData>(std::move(tiled_run_time_data_or_error));

    // This is our way to pass the runtime estimate to the CalculatePriorities()
    // function.
    gpu_performance_model_cache_.Set(
        *producer, *consumer, tiled_run_time_data.runtime_data.exec_time);

    {
      absl::MutexLock lock(&block_level_parameters_cache_mutex_);
      block_level_parameters_cache_[producer][consumer] =
          tiled_run_time_data.block_level_parameters;
    }

    return FusionDecision::Allow();
  }

  FusionDecision CanFuse(HloInstruction* producer, HloInstruction* consumer) {
    // Don't fuse across a root instruction. There are situation when a root
    // instruction is not the last in the computation. Instructions after the
    // root are not necessary dead. They can be inputs to instructions with side
    // effects, like outfeed.
    if (producer == producer->parent()->root_instruction()) {
      return FusionDecision::Forbid(
          "not fusing into the output of the root instruction");
    }

    if (!IsFusible(*producer)) {
      return FusionDecision::Forbid("the producer is not fusible");
    }

    if (!IsFusible(*consumer)) {
      return FusionDecision::Forbid("the consumer is not fusible");
    }

    // Fusing with Triton is our preferred choice. If the producer-consumer
    // fusion is supported by Triton and all necessary flags are enabled, the
    // result will be a Triton fusion. If either `producer` or `consumer` is
    // already a Triton fusion, we can fuse only if the result will also be a
    // Triton fusion.
    //
    // Otherwise, we'll check if the fusion is supported by the emitter.
    FusionDecision can_fuse_triton = CanFuseTriton(producer, consumer);
    if (IsGenericTritonFusion(*producer) || IsGenericTritonFusion(*consumer) ||
        can_fuse_triton) {
      return can_fuse_triton;
    }

    if (HloPredicateIsOp<HloOpcode::kBitcast>(consumer)) {
      return FusionDecision::Forbid(
          "not fusing into a single bitcast as consumer");
    }

    // Scatter is special as it has no elemental version but is still input
    // fusible. Block attempts to create scatter fusions we can't codegen.
    if (auto can_fuse = CanEmitInputFusedScatter(*producer, *consumer);
        !can_fuse) {
      return can_fuse;
    }

    // Avoid fusing reduce into reduce. Our cost model doesn't currently
    // understand this case due to a lack of tiling analysis.
    // TODO(b/312200883): Remove this.
    auto contains_significant_reduce = [&](const HloInstruction* instr) {
      auto fusion = HloFusionAdaptor::ForInstruction(instr);
      return HloAnyOf(*fusion, [](auto node) {
        if (!(node.opcode() == HloOpcode::kReduce && node.shape().IsArray())) {
          return false;
        }

        int64_t reduction_size =
            ShapeUtil::ElementsIn(node.instruction().operand(0)->shape()) /
            ShapeUtil::ElementsIn(node.shape());

        // Small reductions are emitted using the elemental emitter anyway.
        return reduction_size >= 16;
      });
    };
    if (contains_significant_reduce(producer) &&
        contains_significant_reduce(consumer)) {
      return FusionDecision::Forbid(
          "both the producer and the consumer contain a reduce");
    }

    // Avoid doing fusions into the output of an "input" fusion when it would
    // switch it to the loop emitter. This often occurs during epilog fusion for
    // reductions, which suffer from limited emitter support.
    // TODO(b/312686229): Cost model should handle this.
    const auto& analysis = fusion_analysis_cache_.Get(*producer);
    if (analysis.GetEmitterFusionKind() ==
        HloFusionAnalysis::EmitterFusionKind::kReduction) {
      const auto& analysis_fused =
          fusion_analysis_cache_.Get(*producer, *consumer);
      if (analysis_fused.GetEmitterFusionKind() ==
          HloFusionAnalysis::EmitterFusionKind::kLoop) {
        return FusionDecision::Forbid(
            "fusion into output of a reduce fusion would create a loop fusion");
      }
    }

    // Avoid cases where we'd create a fusion that hit limitations in ptxas.
    // Would be nice to model this with cost instead.
    if (auto fits_budget = FusionFitsInBudget(
            *consumer, *producer, *device_info_,
            /*is_consumer_producer_fusion=*/true, &fusion_info_cache_);
        !fits_budget) {
      return fits_budget;
    }

    // Also check that our emitter can handle the fusion node. We currently can
    // have exponential time/memory requirements for emitting certain fusion
    // kernels, in which case we don't want to fuse.
    // TODO(b/119692968): Remove this once we have fixed our fusion emitter.
    if (cost_analysis_.ProducerConsumerMergedTooLarge(*producer, *consumer)) {
      return FusionDecision::Forbid(
          "the fusion would result in an overly large code duplication");
    }

    return InstructionFusion::ShouldFuseInPlaceOp(producer, consumer,
                                                  std::nullopt);
  }

  FusionDecision CanFuseCached(HloInstruction* producer,
                               HloInstruction* consumer) {
    {
      absl::MutexLock lock(&can_fuse_cache_mutex_);
      auto& producer_cache = can_fuse_cache_[producer];

      auto it = producer_cache.find(consumer);
      if (it != producer_cache.end()) {
        return it->second;
      }
    }
    auto fusion_decision = CanFuse(producer, consumer);

    // The lock is required, because writing to a flat_hash_map is not
    // thread-safe even for different keys. We never call this computation
    // concurrently for the same producer, so it's guaranteed that we don't
    // override any value.
    {
      absl::MutexLock lock(&can_fuse_cache_mutex_);
      can_fuse_cache_[producer].insert_or_assign(consumer, fusion_decision);
    }

    return fusion_decision;
  }

  FusionDecision CanFuseWithAllNonBitcastUsers(HloInstruction* producer) {
    if (producer->users().empty()) {
      return FusionDecision::Forbid("No users to fuse");
    }

    bool has_non_bitcast_user = false;
    for (const auto& user : producer->users()) {
      if (HloPredicateIsOp<HloOpcode::kBitcast>(user)) {
        continue;
      }
      has_non_bitcast_user = true;
      if (auto fusion_decision = CanFuseCached(producer, user);
          !fusion_decision) {
        VLOG(10) << "Cannot fuse " << producer->name() << " with "
                 << user->name() << ", because: " << fusion_decision.Explain();
        return fusion_decision;
      }
    }
    if (!has_non_bitcast_user) {
      return FusionDecision::Forbid(
          "not fusing because there are only bitcast users");
    }
    return FusionDecision::Allow();
  }

  // Store computation for cost analysis.
  HloComputation* computation_;

  const se::DeviceDescription* device_info_;

  // Cost Analysis that is used to estimate the cost of a fusion.
  GpuHloCostAnalysis cost_analysis_;

  // Performance model that is used to estimate the run time of a fusion.
  GpuPerformanceModelWithIndexingAnalysis gpu_indexing_performance_model_;

  // The priority queue of producers, implemented as an ordered map, where a
  // key is a pair: the first element is the priority and the second element is
  // the unique ID of the instruction to break ties.
  using PriorityQueue = std::map<std::pair<Priority, int>, HloInstruction*>;
  PriorityQueue producer_priority_queue_;

  // A reverse map that helps find an instruction in the priority queue.
  absl::flat_hash_map<HloInstruction*, PriorityQueue::iterator> reverse_map_;

  // The current producer being visited.
  HloInstruction* current_producer_;

  // The current consumers being visited.
  std::vector<HloInstruction*> current_consumers_;

  // The set of producers whose priorities need to be updated. Their
  // priorities are changed because their neighbors got fused, but we delay
  // the priority updates until current_consumers_ becomes empty. This is to
  // avoid recomputing priorities multiple times before we dequeue a new
  // producer.
  absl::flat_hash_set<HloInstruction*> to_update_priority_;
  absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>
      operands_to_new_consumers_;
  absl::flat_hash_map<HloInstruction*, GpuPerformanceModel::RunTimes>
      operands_to_removed_consumers_runtimes_;
  // Proto with structured logs of fusion decisions. Used only for debugging. If
  // null, logging is disabled.
  FusionProcessDumpProto* fusion_process_dump_;
  absl::Mutex fusion_process_dump_mutex_;

  tsl::thread::ThreadPool* thread_pool_;

  mlir::MLIRContext* mlir_context_;

  HloFusionAnalysisCache& fusion_analysis_cache_;

  FusionDeduplicationCache& fusion_deduplication_cache_;
  absl::Mutex fusion_deduplication_cache_mutex_;

  // Caches result of can_fuse for a (producer, consumer) pair. A cache entry is
  // invalidated if producer or consumer is modified.
  absl::flat_hash_map<
      const HloInstruction*,
      absl::flat_hash_map<const HloInstruction*, FusionDecision>>
      can_fuse_cache_;
  absl::Mutex can_fuse_cache_mutex_;

  // Caches block level parameters for a (producer, consumer) pair. A cache
  // entry is invalidated if producer or consumer is modified.
  absl::flat_hash_map<
      const HloInstruction*,
      absl::flat_hash_map<const HloInstruction*, BlockLevelParameters>>
      block_level_parameters_cache_;
  absl::Mutex block_level_parameters_cache_mutex_;

  absl::flat_hash_map<FusionDeduplicationCache::FusionId,
                      TiledRunTimeDataOrError>
      tiled_run_time_data_cache_;
  absl::Mutex tiled_run_time_data_cache_mutex_;

  GpuPerformanceModelCache gpu_performance_model_cache_;

  // Cache for `FusionFitsInBudget` to avoid recomputing expensive properties
  // like shared memory usage or number of unnested reductions of fusion nodes.
  FusionInfoCache fusion_info_cache_;

  // If true, redirect all fusion decisions to Triton fusion.
  bool triton_heroless_fusion_enabled_;

  bool dump_fusion_visualization_;
};

}  // namespace

// Return true, if instr is a small constant.
//
// There is not single definition for what is a small constant in XLA.
// IrEmitterContext::emit_constant treats as small only constants of 1 element.
// HloPrintOptions::print_large_constants is effective for constants larger
// than 10 elements.
//
// This function matches the emitter logic.
bool IsSmallConstant(const HloInstruction* instr) {
  return HloPredicateIsOp<HloOpcode::kConstant>(instr) &&
         instr->shape().IsArray() && ShapeUtil::ElementsIn(instr->shape()) <= 1;
}

bool PriorityFusion::ConsumeFuel(HloInstruction* producer,
                                 HloInstruction* consumer) {
  return xla::ConsumeFuel(name(), /*ran_out_of_fuel_msg=*/[&] {
    return absl::StrFormat("Not fusing producer %s with consumer %s",
                           producer->name(), consumer->name());
  });
};

FusionDecision PriorityFusion::CanFuseConstant(const HloInstruction* constant,
                                               const HloInstruction* user) {
  // If user is a scatter, verify that we can fuse the constant correctly.
  if (auto fusion_decision = CanEmitInputFusedScatter(*constant, *user);
      !fusion_decision) {
    return fusion_decision;
  }

  // If user is a Triton fusion, verify that the constant is supported
  // by Triton.
  //
  // Note: `IsFusible` should not be used for Triton fusions. Generally,
  // `IsFusible` returns `false` for Triton fusions, because Triton fusions have
  // kCustom fusion kind, but sometimes `IsFusible` will return `true` if the
  // fusion contains only elementwise instructions.
  // We can always fuse a producer into Triton fusions if the producer is
  // supported by Triton, so it's enough to check if the constant is supported.
  if (IsGenericTritonFusion(*user)) {
    return IsTritonSupportedInstruction(*constant,
                                        device_info_.gpu_compute_capability());
  }

  // Verify that the user is fusible.
  if (!IsFusible(*user)) {
    return FusionDecision::Forbid("User is not fusible");
  }

  return FusionDecision::Allow();
}

absl::StatusOr<bool> PriorityFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool dump_enabled =
      DumpingEnabledForHloPass(name(), module->config().debug_options());
  if (dump_enabled) {
    fusion_process_dump_ = std::make_unique<FusionProcessDumpProto>();
    *fusion_process_dump_->mutable_gpu_device_info() =
        device_info_.ToGpuProto();
  }

  // Compute the computations within which more fusion is possible.
  auto fusible_computations =
      GetFusibleComputations(*module, execution_threads);

  // Appends ".0" suffix to all instructions.
  //
  // Every time an instruction is duplicated, the last integer suffix is
  // incremented.
  // Before: broadcast.123 -> broadcast.124
  // After: broadcast.123.0 -> broadcast.123.1
  //
  // With this modification it will be easier to match instructions before and
  // after fusion passes, because they will have the same unique prefix. Names
  // are not used in the pipeline, but it makes debugging much easier.
  for (auto* computation : fusible_computations) {
    for (auto* instruction : computation->instructions()) {
      module->SetAndUniquifyInstrName(instruction,
                                      absl::StrCat(instruction->name(), ".0"));
    }
  }

  if (dump_enabled) {
    fusion_process_dump_->set_hlo_module_before_fusion(
        module->ToString(HloPrintOptions::ShortParsable()));
  }

  bool triton_heroless_fusion_enabled =
      module->config()
          .debug_options()
          .xla_gpu_experimental_enable_triton_heroless_priority_fusion();

  FusionDeduplicationCache fusion_deduplication_cache =
      FusionDeduplicationCache::Create(*module, IsFusible);

  bool changed = false;
  for (auto* computation : fusible_computations) {
    CHECK(!computation->IsFusionComputation());

    auto fusion_queue = std::make_unique<PriorityFusionQueue>(
        computation, cost_analysis_options_, &device_info_,
        fusion_process_dump_.get(), thread_pool_, &mlir_context_,
        fusion_analysis_cache_, fusion_deduplication_cache,
        triton_heroless_fusion_enabled);

    while (fusion_queue->DequeueNextProducer()) {
      auto producer = fusion_queue->current_producer();

      absl::flat_hash_map<const HloInstruction*, BlockLevelParameters>
          block_level_parameters_map =
              fusion_queue->GetBlockLevelParametersMap(producer);

      for (auto* consumer : fusion_queue->current_consumers()) {
        // Don't fuse into single bitcasts. We ignore them in the check
        // CanFuseWithAllNonBitcastUsers(), so we need to check it here.
        if (HloPredicateIsOp<HloOpcode::kBitcast>(consumer)) {
          continue;
        }
        if (!ConsumeFuel(producer, consumer)) continue;

        VLOG(5) << "next: " << consumer->name() << "(" << consumer << ") + "
                << producer->name() << "(" << producer << ")";

        int64_t consumer_operand_index = consumer->operand_index(producer);

        fusion_queue->PreFusion(producer, consumer);
        auto fusion_instruction = Fuse(producer, consumer);
        fusion_deduplication_cache.UpdateFusedInstructionId(
            fusion_instruction, producer, consumer, consumer_operand_index);
        fusion_queue->OnFusingInstruction(fusion_instruction, producer,
                                          consumer);

        auto backend_config_it = block_level_parameters_map.find(consumer);
        if (backend_config_it != block_level_parameters_map.end()) {
          TF_RETURN_IF_ERROR(fusion_instruction->set_backend_config(
              GetTritonGpuBackendConfig(backend_config_it->second)));
          fusion_instruction->set_fusion_kind(
              HloInstruction::FusionKind::kCustom);
        }

        changed = true;
      }

      fusion_queue->ComputeRuntimesOfRemovedConsumers();
      if (producer->user_count() == 0) {
        fusion_queue->InvalidateCaches(producer);
        producer->DetachFromOperandsAndUsers();
        fusion_queue->RemoveInstruction(producer);
        // Remove from computation.
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(producer));
      }

      for (auto* consumer : fusion_queue->current_consumers()) {
        fusion_queue->InvalidateCaches(consumer);
      }
      TF_RETURN_IF_ERROR(fusion_queue->UpdatePriorities());
    }

    // Fuse all constants.
    std::vector<HloInstruction*> constants;
    for (auto* instruction : computation->instructions()) {
      // Small constants should be fused, because they can be folded and
      // codegened efficiently.
      // Fusing large constants doesn't give much benefits, because they're
      // treated like parameters and read from global memory anyway. Fusion
      // and duplication of large constants can, however, cause problems if we
      // want to dump hlo and parse back, because in that case duplicated
      // constants will be filled with different data.
      if (IsSmallConstant(instruction)) {
        constants.push_back(instruction);
      }
    }

    for (auto* constant : constants) {
      auto users = constant->users();
      for (auto* user : users) {
        if (CanFuseConstant(constant, user)) {
          Fuse(constant, user);
          changed = true;
        }
      }
    }
  }

  // FusionAnalysis cache uses unique_id as key. IDs are only unique inside one
  // module. It's important to fully clear the cache if the same instance of the
  // pass will be called on a different module.
  fusion_analysis_cache_.Clear();

  if (dump_enabled) {
    DumpPerModuleProtobufToFile(*module, *fusion_process_dump_,
                                module->config().debug_options(),
                                "priority_fusion_dump");
  }

  return changed;
}

HloInstruction::FusionKind PriorityFusion::ChooseKind(
    const HloInstruction* producer, const HloInstruction* consumer) {
  // Derive kInput/kLoop fusion kinds from fusion analysis. This shouldn't
  // matter but some passes downstream still query these instead of fusion
  // analysis.
  const auto& analysis = fusion_analysis_cache_.Get(*producer, *consumer);
  switch (analysis.GetEmitterFusionKind()) {
    case HloFusionAnalysis::EmitterFusionKind::kLoop:
      return HloInstruction::FusionKind::kLoop;
    case HloFusionAnalysis::EmitterFusionKind::kTriton:
    case HloFusionAnalysis::EmitterFusionKind::kCustomFusion:
    case HloFusionAnalysis::EmitterFusionKind::kCuDnn:
      return HloInstruction::FusionKind::kCustom;
    case HloFusionAnalysis::EmitterFusionKind::kConcatenate:
    case HloFusionAnalysis::EmitterFusionKind::kReduction:
    case HloFusionAnalysis::EmitterFusionKind::kTranspose:
    case HloFusionAnalysis::EmitterFusionKind::kInputSlices:
    case HloFusionAnalysis::EmitterFusionKind::kScatter:
      return HloInstruction::FusionKind::kInput;
  }
}

HloInstruction* PriorityFusion::Fuse(HloInstruction* producer,
                                     HloInstruction* consumer) {
  VLOG(2) << "Fusing " << producer->ToString() << " into "
          << consumer->ToString();

  HloComputation* computation = consumer->parent();
  auto kind = ChooseKind(producer, consumer);
  HloInstruction* fusion_instruction = consumer;

  if (HloPredicateIsNotOp<HloOpcode::kFusion>(fusion_instruction)) {
    fusion_instruction = computation->AddInstruction(
        HloInstruction::CreateFusion(consumer->shape(), kind, consumer));
    TF_CHECK_OK(computation->ReplaceInstruction(consumer, fusion_instruction));
  } else if (kind != fusion_instruction->fusion_kind()) {
    fusion_instruction->set_fusion_kind(kind);
  }

  fusion_instruction->set_called_computations_execution_thread(
      computation->execution_thread(),
      /*skip_async_execution_thread_overwrite=*/false);

  if (HloPredicateIsOp<HloOpcode::kFusion>(producer)) {
    fusion_instruction->MergeFusionInstruction(producer);
  } else {
    fusion_instruction->FuseInstruction(producer);
  }

  if (fusion_instruction != consumer) {
    VLOG(2) << "       created new fusion: " << fusion_instruction->ToString();
  }

  return fusion_instruction;
}

}  // namespace gpu
}  // namespace xla
