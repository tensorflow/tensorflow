/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.h"

#include <deque>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/latency_hiding_scheduler.h"

namespace xla {
namespace gpu {

namespace {

bool ShouldScheduleAsEarlyAsPossible(const HloInstruction& instr) {
  switch (instr.opcode()) {
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kCollectivePermuteStart:
      return true;
    case HloOpcode::kCustomCall:
      return static_cast<const HloCustomCallInstruction&>(instr)
                 .custom_call_schedule() ==
             CustomCallSchedule::SCHEDULE_EARLIEST;
    default:
      return false;
  }
}

bool ShouldScheduleSuccessor(const HloInstruction& sussessor,
                             const HloPredicate& is_scheduled) {
  return ShouldScheduleAsEarlyAsPossible(sussessor) &&
         absl::c_all_of(sussessor.operands(), is_scheduled) &&
         absl::c_all_of(sussessor.control_predecessors(), is_scheduled);
}

bool ShouldScheduleAsLateAsPossible(const HloInstruction& instr) {
  switch (instr.opcode()) {
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kCollectivePermuteDone:
      return true;
    case HloOpcode::kCustomCall:
      return static_cast<const HloCustomCallInstruction&>(instr)
                 .custom_call_schedule() == CustomCallSchedule::SCHEDULE_LATEST;
    default:
      return false;
  }
}

bool ShouldSchedulePredecessor(const HloInstruction& predecessor,
                               const HloPredicate& is_scheduled) {
  return ShouldScheduleAsLateAsPossible(predecessor) &&
         absl::c_all_of(predecessor.users(), is_scheduled) &&
         absl::c_all_of(predecessor.control_successors(), is_scheduled);
}

// Schedules certain ops as early or late as possible. This supports a
// custom-call use case, where a logical operation is lowered into two HLOs
// (e.g., PerformX and PerformXDone). We utilize this mechanism to either hide
// host latencies between the pair of the custom-calls or more accurately
// identify the def-use relationship of the two calls (typically PerformX is
// scheduled right after all of its producers have been scheduled and
// PerformXDone is scheduled right before its first consumer.)
HloInstructionSequence PostprocessorToScheduleAsEarlyOrLateAsPossible(
    const HloInstructionSequence& input) {
  std::vector<HloInstruction*> earliest_scheduled;
  {
    absl::flat_hash_set<HloInstruction*> scheduled;
    auto is_scheduled = [&](const HloInstruction* instr) -> bool {
      return scheduled.contains(instr);
    };
    auto add_to_schedule = [&](HloInstruction* instr) {
      earliest_scheduled.push_back(instr);
      scheduled.insert(instr);
    };
    for (HloInstruction* instr : input.instructions()) {
      if (is_scheduled(instr)) {
        continue;
      }

      add_to_schedule(instr);

      // Schedule any successor that should be scheduled as early as possible if
      // all of its producers and control_predecessors have been scheduled.
      for (HloInstruction* user : instr->users()) {
        if (ShouldScheduleSuccessor(*user, is_scheduled)) {
          add_to_schedule(user);
        }
      }
      for (HloInstruction* successor : instr->control_successors()) {
        if (ShouldScheduleSuccessor(*successor, is_scheduled)) {
          add_to_schedule(successor);
        }
      }
    }
  }

  std::deque<HloInstruction*> latest_scheduled;
  {
    absl::flat_hash_set<HloInstruction*> scheduled;
    auto is_scheduled = [&](const HloInstruction* instr) -> bool {
      return scheduled.contains(instr);
    };
    auto add_to_schedule = [&](HloInstruction* instr) {
      latest_scheduled.push_front(instr);
      scheduled.insert(instr);
    };
    for (auto it = earliest_scheduled.rbegin(); it != earliest_scheduled.rend();
         it++) {
      if (is_scheduled(*it)) {
        continue;
      }

      add_to_schedule(*it);

      // Schedule any predecessor that should be scheduled as late as possible
      // if all of its users and control_successors have been scheduled.
      for (HloInstruction* operand : (*it)->operands()) {
        if (ShouldSchedulePredecessor(*operand, is_scheduled)) {
          add_to_schedule(operand);
        }
      }
      for (HloInstruction* predecessor : (*it)->control_predecessors()) {
        if (ShouldSchedulePredecessor(*predecessor, is_scheduled)) {
          add_to_schedule(predecessor);
        }
      }
    }
  }

  HloInstructionSequence result;
  absl::c_for_each(latest_scheduled,
                   [&](HloInstruction* i) { result.push_back(i); });
  return result;
}

StatusOr<HloSchedule> ScheduleGpuModuleWithMemoryScheduler(
    const HloModule* module, int64_t pointer_size, bool enable_post_processor) {
  MemorySchedulerPostprocessor post_processor =
      enable_post_processor ? PostprocessorToScheduleAsEarlyOrLateAsPossible
                            : nullptr;
  return ScheduleModule(
      module,
      [pointer_size](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), pointer_size);
      },
      ComputationSchedulerToModuleScheduler(DefaultMemoryScheduler,
                                            post_processor));
}

// Latency hiding scheduler support.

SchedulerConfig GetSchedulerConfig(const GpuDeviceInfo& gpu_info) {
  SchedulerConfig config;
  config.all_reduce_overlap_limit = 1;
  config.collective_permute_overlap_limit = 1;
  config.use_real_cost_model = false;
  config.aggressive_scheduling_policies = true;

  // Assume 75% of the total device memory is available for XLA.
  config.memory_limit = gpu_info.device_memory_size * 0.75;
  return config;
}

// GPU specific resources for latency hiding scheduler.
enum class GpuResourceType {
  kGpuAsyncStream = 0,  // The async stream for collectives.
  kNumTargetResources = 1,
};

// GPU async tracker maps all collectives onto a async stream resource.
class GpuAsyncTracker : public AsyncTracker {
 public:
  explicit GpuAsyncTracker(const SchedulerConfig& config)
      : AsyncTracker(config) {}

  ResourcesVector GetResourcesFromInstruction(
      const HloInstruction& instr) const override {
    CanonicalAsyncOp op = GetCanonicalAsyncOp(instr);
    if (op.outer == HloOpcode::kAsyncStart ||
        op.outer == HloOpcode::kAsyncDone) {
      ResourceUsageType usage = op.outer == HloOpcode::kAsyncStart
                                    ? ResourceUsageType::kResourceRelease
                                    : ResourceUsageType::kResourceOccupy;

      const int64_t gpu_stream_resource =
          GetFirstTargetDefinedResource() +
          static_cast<int64_t>(GpuResourceType::kGpuAsyncStream);
      return {std::make_pair(gpu_stream_resource, usage)};
    }
    return AsyncTracker::GetResourcesFromInstruction(instr);
  }

  int64_t GetNumTargetDefinedResources() const override {
    return static_cast<int64_t>(GpuResourceType::kNumTargetResources);
  };

  // Returns how many instructions using the given resource_type we can overlap
  int64_t GetNumAvailableResources(int64_t resource_type) const override {
    const int64_t first_target_resource = GetFirstTargetDefinedResource();
    if (resource_type < first_target_resource) {
      return AsyncTracker::GetNumAvailableResources(resource_type);
    }
    CHECK_EQ(resource_type,
             first_target_resource +
                 static_cast<int64_t>(GpuResourceType::kGpuAsyncStream));

    // We will allow upto 1 outstanding collective on the async stream. This
    // controls the number of collectives in flight in the schedule (a
    // collective is in flight if the start is issued but not done). As an
    // example, with 1, LHS will generate the schedule: s0,e0,s1,e1, i.e., s1
    // is not scheduled until e0 is scheduled. With 2, the scheduler can
    // schedule s0,s1,e0,e1, because it assumes that the 2 instances of the
    // resources do not interfere with each other. If we do want to support > 1
    // async stream, we can increase this number and then do a post-pass on the
    // scheduled code to assign async stream-id to collectives (and actually
    // support > 1 async stream in the runtime).
    return 1;
  }

  absl::string_view GetResourceName(int64_t resource_type) const override {
    const int64_t first_target_resource = GetFirstTargetDefinedResource();
    if (resource_type < first_target_resource) {
      return AsyncTracker::GetResourceName(resource_type);
    }
    CHECK_LE(resource_type,
             first_target_resource + GetNumTargetDefinedResources());
    switch (resource_type - first_target_resource) {
      case static_cast<int64_t>(GpuResourceType::kGpuAsyncStream):
        return "kGpuAsyncStream";
      default:
        return "kUnsupportedResource";
    }
  }

  ResourceHazardType GetResourceHazardType(
      int64_t resource_type) const override {
    const int64_t first_target_resource = GetFirstTargetDefinedResource();
    if (resource_type < first_target_resource) {
      return AsyncTracker::GetResourceHazardType(resource_type);
    }
    CHECK_LE(resource_type,
             first_target_resource + GetNumTargetDefinedResources());
    return ResourceHazardType::kUnshareable;
  }
};

class GpuLatencyEstimator : public ApproximateLatencyEstimator {
 public:
  TimeCost NodeCost(const HloInstruction* instr) const override {
    // Consider cublas/cuddn/softmax custom calls as medium cost. Since the
    // latency between async-start and async-done is 5000 and cost of each
    // custom call is 1000, the LHS will try to schedule approximately 5 of
    // these in between each start/end pair.
    if (instr->opcode() == HloOpcode::kCustomCall) {
      if (IsCublasGemm(*instr) || IsCustomCallToDnnConvolution(*instr)) {
        return ApproximateLatencyEstimator::kMediumCost;
      }
      // consider other custom calls as medium cost for now. Keeping the case
      // explicitly separate for further tuning.
      return ApproximateLatencyEstimator::kMediumCost;
    }
    return ApproximateLatencyEstimator::NodeCost(instr);
  }
};

}  // end namespace

int64_t GetSizeOfShape(const Shape& shape, int pointer_size) {
  int64_t size = ShapeUtil::ByteSizeOf(shape, pointer_size);
  if (shape.is_static() || shape.IsTuple()) {
    return size;
  }
  // Each dynamic dimension size is represented as a S32.
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions_size();
  return size + metadata_size;
}

Status ScheduleGpuModule(HloModule* module, int64_t pointer_size,
                         const GpuDeviceInfo& gpu_info) {
  const bool enable_latency_hiding_scheduler =
      module->config()
          .debug_options()
          .xla_gpu_enable_latency_hiding_scheduler();
  TF_ASSIGN_OR_RETURN(
      HloSchedule schedule,
      ScheduleGpuModuleWithMemoryScheduler(module, pointer_size,
                                           !enable_latency_hiding_scheduler));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));

  if (!enable_latency_hiding_scheduler) {
    return OkStatus();
  }
  SchedulerConfig config = GetSchedulerConfig(gpu_info);
  auto latency_estimator = std::make_unique<GpuLatencyEstimator>();
  auto async_tracker = [&]() -> std::unique_ptr<AsyncTracker> {
    return module->config()
                   .debug_options()
                   .xla_gpu_lhs_enable_gpu_async_tracker()
               ? std::make_unique<GpuAsyncTracker>(config)
               : std::make_unique<AsyncTracker>(config);
  }();

  auto shape_size_in_bytes = [pointer_size](const Shape& shape) {
    return GetSizeOfShape(shape, pointer_size);
  };
  HloPassPipeline pipeline("latency-hiding-scheduler");
  auto scheduler_core = std::make_unique<DefaultSchedulerCore>(
      shape_size_in_bytes, async_tracker.get(), latency_estimator.get(),
      config);

  pipeline.AddPass<LatencyHidingScheduler>(
      std::move(latency_estimator), std::move(async_tracker),
      std::move(scheduler_core), shape_size_in_bytes);

  TF_RETURN_IF_ERROR(pipeline.Run(module).status());
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
