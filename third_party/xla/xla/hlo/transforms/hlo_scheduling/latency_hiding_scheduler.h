/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_HLO_SCHEDULING_LATENCY_HIDING_SCHEDULER_H_
#define XLA_HLO_TRANSFORMS_HLO_SCHEDULING_LATENCY_HIDING_SCHEDULER_H_

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/transforms/hlo_scheduling/approximate_latency_estimator.h"  // IWYU pragma: export
#include "xla/hlo/transforms/hlo_scheduling/async_tracker.h"  // IWYU pragma: export
#include "xla/hlo/transforms/hlo_scheduling/hlo_schedule_graph.h"  // IWYU pragma: export
#include "xla/hlo/transforms/hlo_scheduling/latency_estimator.h"  // IWYU pragma: export
#include "xla/hlo/transforms/hlo_scheduling/scheduler_config.h"  // IWYU pragma: export
#include "xla/hlo/transforms/hlo_scheduling/scheduler_core.h"  // IWYU pragma: export
#include "xla/layout.h"
#include "xla/map_util.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_value.h"
#include "xla/shape_util.h"
#include "xla/xla.pb.h"

namespace xla {

inline bool IsNopInstruction(HloOpcode op, const HloInstruction& hlo) {
  return op == HloOpcode::kGetTupleElement || op == HloOpcode::kBitcast ||
         op == HloOpcode::kConstant || op == HloOpcode::kParameter ||
         op == HloOpcode::kBroadcast || op == HloOpcode::kIota ||
         hlo.IsEffectiveBitcast(op) ||
         (op == HloOpcode::kTuple && hlo.user_count() == 1 &&
          hlo.users().front()->opcode() == HloOpcode::kWhile);
}

class ModulePressureState;

// Tracks data about HloBuffers like where the first definition is in the
// original schedule and caches the buffer size (as Target::ShapeSize()) is
// expensive.
class BufferInfoTracker {
 public:
  struct ValueInfo {
    const HloBuffer* value = nullptr;
    const HloInstruction* first_definition = nullptr;
    const HloInstruction* last_use = nullptr;
    int64_t buffer_size = 0;

    // Precomputed value of
    //     value->values()[0]->shape().has_layout() &&
    //     (value->values()[0]->shape().layout().memory_space() !=
    //      kDefaultMemorySpace)
    // This expression is invoked repeatedly and is responsible for many cache
    // misses.
    bool non_default_memory_space_layout = false;
    std::vector<const HloInstruction*> transitively_defining_calls;
  };
  BufferInfoTracker(const HloModule* module,
                    const HloAliasAnalysis* alias_analysis,
                    const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes);
  static ValueInfo CreateBufferInfo(
      const HloBuffer* value, const HloInstruction* first_definition,
      const HloInstruction* last_use,
      const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes) {
    const auto& shape = value->values()[0]->shape();
    const bool non_default_memory_space_layout =
        (shape.has_layout() &&
         (shape.layout().memory_space() != Layout::kDefaultMemorySpace));
    return ValueInfo{
        /*value=*/value,
        /*first_definition=*/first_definition,
        /*last_use=*/last_use,
        /*buffer_size=*/shape_size_bytes(shape),
        /*non_default_memory_space_layout=*/non_default_memory_space_layout,
    };
  }
  const ValueInfo& GetBufferInfo(HloBuffer::Id id) const {
    return buffer_infos_[id];
  }

 private:
  std::vector<ValueInfo> buffer_infos_;
};

// Types used for memory pressure tracking (shared between
// MemoryPressureMetadata and MemoryPressureTracker)
struct MemoryPressureState {
  int64_t memory_peak = 0;
  absl::flat_hash_set<HloBuffer::Id> live_ids_at_bottom;
};

using LiveBufferSet = absl::flat_hash_set<HloBuffer::Id>;

// Holds precomputed metadata for memory pressure tracking.
class MemoryPressureMetadata {
 public:
  MemoryPressureMetadata(
      const HloAliasAnalysis* hlo_alias_analysis,
      const BufferInfoTracker& buffer_tracker,
      const absl::flat_hash_map<const HloComputation*, MemoryPressureState>&
          pressure_state_cache,
      bool top_down_scheduling = false)
      : hlo_alias_analysis_(hlo_alias_analysis),
        buffer_tracker_(buffer_tracker),
        pressure_state_cache_(pressure_state_cache),
        top_down_scheduling_(top_down_scheduling) {}
  // Initialize object to be ready to start tracking of computation, this
  // should be called once per computation.
  void Initialize(const HloComputation* computation);

  int32_t GetInstructionId(const HloInstruction* i) const {
    auto it = instruction_ids_.find(i);
    CHECK(it != instruction_ids_.end());
    return it->second;
  }

  absl::Span<const HloBuffer::Id> allocated_buffer_ids(
      const HloInstruction* i) const {
    auto it = instruction_ids_.find(i);
    CHECK(it != instruction_ids_.end());
    NodeAllocReleaseSpan s = alloc_release_spans_[it->second];
    return absl::MakeSpan(alloc_release_ids_).subspan(s.start, s.num_alloc);
  }

  absl::Span<const HloBuffer::Id> released_buffer_ids(
      const HloInstruction* i) const {
    auto it = instruction_ids_.find(i);
    CHECK(it != instruction_ids_.end());
    NodeAllocReleaseSpan s = alloc_release_spans_[it->second];
    return absl::MakeSpan(alloc_release_ids_)
        .subspan(s.start + s.num_alloc, s.num_release);
  }

  int64_t num_buffers() const {
    return hlo_alias_analysis_->buffers().back().id() + 1;
  }

  const BufferInfoTracker& buffer_tracker() const { return buffer_tracker_; }
  const absl::flat_hash_map<const HloComputation*, MemoryPressureState>&
  pressure_state_cache() const {
    return pressure_state_cache_;
  }

 private:
  // Append to *dst the list of buffer ids allocated by instruction whose
  // memory usage should be tracked. Returns number of ids added.
  int32_t ComputeBufferAllocations(const HloInstruction* instruction,
                                   std::vector<HloBuffer::Id>* dst);

  // Append to *dst the list of buffer ids released by instruction whose
  // memory usage should be tracked. Returns number of ids added.
  int32_t ComputeBufferReleases(const HloInstruction* instruction,
                                std::vector<HloBuffer::Id>* dst);

  static bool ShouldSkipBufferAllocations(
      const HloInstruction* instruction, const ShapeIndex& idx,
      const HloInstruction* first_definition) {
    // Make GetTupleElement/kBitcast make alive only the tuple pointer if not
    // array shape.
    if ((instruction->opcode() == HloOpcode::kGetTupleElement ||
         instruction->opcode() == HloOpcode::kBitcast) &&
        !idx.empty()) {
      return true;
    }
    // Skip entry computation parameters because their memory usage is already
    // accounted for.
    if (first_definition->opcode() == HloOpcode::kParameter &&
        first_definition->parent()->IsEntryComputation()) {
      return true;
    }
    return false;
  }
  static bool ShouldSkipBufferReleases(const HloInstruction* instruction) {
    // Do not release parameter buffers as they are still in use by the caller.
    if (instruction->opcode() == HloOpcode::kParameter) {
      return true;
    }
    return false;
  }
  bool InstructionDefinesValue(const HloInstruction* instruction,
                               const HloValue* value) const;
  bool InstructionFirstDefinesBuffer(
      const HloInstruction* instruction,
      const BufferInfoTracker::ValueInfo& buffer_value_info) const;
  bool InstructionTransitivelyDefines(
      const HloInstruction* instruction,
      const BufferInfoTracker::ValueInfo& buffer_value_info) const;
  const HloAliasAnalysis* hlo_alias_analysis_;

  // Mapping from instruction to dense id.
  absl::flat_hash_map<const HloInstruction*, int32_t> instruction_ids_;

  // Combined vector of buffers allocated/released by each node.
  // See allocated_buffer_ids() and released_buffer_ids().
  std::vector<HloBuffer::Id> alloc_release_ids_;

  // Information kept per node that identifies allocated/released buffers.
  struct NodeAllocReleaseSpan {
    // Allocated buffers in alloc_release_ids_[start,start+num_alloc).
    // Released buffers in
    // alloc_release_ids_[start+num_alloc,start+num_alloc+num_release)
    uint32_t start;
    uint32_t num_alloc;
    uint32_t num_release;
  };

  // Mapping from dense instruction id to span information within
  // alloc_release_ids_.
  std::vector<NodeAllocReleaseSpan> alloc_release_spans_;

  const BufferInfoTracker& buffer_tracker_;
  // Cache of buffer objects defined that are output of instructions.
  absl::flat_hash_map<
      HloInstruction*,
      std::vector<std::pair<BufferInfoTracker::ValueInfo, ShapeIndex>>>
      output_buffers_;
  // Cache of buffer objects defined that are defined by instructions.
  absl::flat_hash_map<HloInstruction*,
                      std::vector<BufferInfoTracker::ValueInfo>>
      defined_buffers_;
  // Map with pressure_state object for other computations. It's updated by
  // the user of this class.
  const absl::flat_hash_map<const HloComputation*, MemoryPressureState>&
      pressure_state_cache_;
  bool top_down_scheduling_;
};

// Used to track and maintain dynamic memory pressure during scheduling.
class MemoryPressureTracker {
 public:
  // MemoryPressureTracker is used to track the memory pressure of a single
  // computation. It is initialized with the initial live buffers at the bottom
  // of the computation. As the algorithm progresses, the live buffers change
  // and the memory pressure changes.
  // The memory pressure of a computation is the sum of the initial memory
  // pressure and the live memory pressure.
  // The live memory pressure is the difference between the current memory usage
  // and the initial memory pressure.
  //
  // These trackers are intended to be ephemeral objects, used in a single
  // scheduling attempt.
  explicit MemoryPressureTracker(const MemoryPressureMetadata* metadata,
                                 const LiveBufferSet& initial_live_buffers = {})
      : metadata_(metadata),
        live_buffers_(metadata ? metadata->num_buffers() : 0),
        live_memory_usage_(0),
        initial_memory_pressure_(0) {
    if (metadata) {
      Reset(nullptr, initial_live_buffers);
    }
  }

  // Reset the memory pressure tracker to the initialized state.
  void Reset(const HloComputation* /*computation*/,
             const LiveBufferSet& initial_live_buffers);

  // After an instruction is scheduled, update the memory pressure effect on
  // other instructions.
  void UpdateBuffers(const HloInstruction* instruction);
  // Return the memory pressure difference estimation if this instruction was
  // scheduled.
  // Returns a pair of (increase, peak) values.
  // "increase" determines by how much the memory pressure increases or
  // decreases after this instruction is scheduled. "peak" determines what's the
  // peak usage of memory of the computation. The peak can be higher than the
  // total memory increase of the instruction (imagine a computation called by a
  // while loop, the body of the while could use quite some more memory than the
  // amount of memory at the interfaces of the while loop instruction).
  std::pair<int64_t, int64_t> MemoryPressureDifference(
      const HloInstruction* instruction) const;
  absl::flat_hash_set<HloBuffer::Id> live_buffers() const {
    return live_buffers_set_;
  }
  bool BufferIsLive(const HloValue* buffer) const {
    CHECK_LT(buffer->id(), live_buffers_.size());
    return live_buffers_[buffer->id()];
  }
  // Returns the actual memory usage at the current state. It is initial memory
  // + current memory usage inside of the computation.
  int64_t memory_usage() const {
    return live_memory_usage_ + initial_memory_pressure_;
  }
  // Returns the initial memory pressure at the bottom of the computation.
  int64_t initial_memory_pressure() const { return initial_memory_pressure_; }

  // Returns pressure state object for this MemoryPressureTracker object.
  const MemoryPressureState& pressure_state() const { return pressure_state_; }

 private:
  // Metadata for the computation containing schedule-independent information
  // about computation buffers.
  const MemoryPressureMetadata* metadata_;
  // Live buffer presence set. This is used to determine if a buffer is live or
  // not in a fast way. Because this is checked very often in the evaluation
  // function of the scheduler quering the live_buffer_set_ object is too slow.
  // This is much faster in a tight loop. Also we use int8_t explicitly rather
  // than "bool" as "bool" is optimized and bit-packed trading memory for bit
  // extract operations.
  std::vector<int8_t> live_buffers_;
  // Set of live buffer ids.
  LiveBufferSet live_buffers_set_;
  // Current memory usage delta from the initial memory of the computation.
  int64_t live_memory_usage_;
  // Initial memory pressure at the bottom of the computation.
  int64_t initial_memory_pressure_;
  MemoryPressureState pressure_state_;
};

// Module memory pressure state object. Handles and holds all the objects used
// to store information about memory pressure for computations.
// Computes initial pressure state.
class ModulePressureState {
 public:
  using PressureStateMap =
      absl::flat_hash_map<const HloComputation*, MemoryPressureState>;
  ModulePressureState(
      const HloModule* module, const HloAliasAnalysis* hlo_alias_analysis,
      const HloCostAnalysis::ShapeSizeFunction& shape_size_bytes,
      bool top_down_scheduling = false)
      : module_(module),
        hlo_alias_analysis_(hlo_alias_analysis),
        buffer_tracker_(module, hlo_alias_analysis, shape_size_bytes),
        top_down_scheduling_(top_down_scheduling) {}
  // Initializes the pressure states for all computations in the module, only
  // needs to be called once at the beginning of the scheduling process.
  void InitializePressureStates();
  // Resets the pressure states for all computations in the module, must be
  // called to reset the memory pressure trackers after each scheduling attempt.
  void ResetPressureStates();
  bool ComputationIsMemoryTracked(const HloComputation* computation) const {
    return ContainsKey(memory_pressure_states_, computation);
  }
  // Get memory pressure state for a certain computation stored in this class.
  const MemoryPressureState& GetPressureStateForComputation(
      const HloComputation* comp) const {
    auto it = memory_pressure_states_.find(comp);
    CHECK(it != memory_pressure_states_.end())
        << "No state for " << comp->name();
    return it->second;
  }
  // Updates the memory pressure state cache.
  void UpdatePressureStateForComputation(const HloComputation* comp,
                                         MemoryPressureState state) {
    auto [it, inserted] = memory_pressure_states_.insert_or_assign(comp, state);

    if (!inserted) {
      // Rescheduling computation that has already been scheduled
      // can only happen during preference/heuristic rescheduling.
      // Recalculate memory peak.
      memory_peak_ = 0;
      for (auto& memory_state : memory_pressure_states_) {
        memory_peak_ = std::max(memory_peak_, memory_state.second.memory_peak);
      }
    } else {
      memory_peak_ = std::max(memory_peak_, state.memory_peak);
    }
  }
  // Returns the underlying pressure state cache object
  const PressureStateMap& pressure_state_cache() const {
    return memory_pressure_states_;
  }
  // Returns the buffer tracker object.
  const BufferInfoTracker& buffer_tracker() const { return buffer_tracker_; }
  int64_t GetMemoryPeak() { return memory_peak_; }
  void SetMemoryPeak(int64_t peak) { memory_peak_ = peak; }
  const MemoryPressureMetadata* GetOrCreatePressureMetadata(
      const HloComputation* comp) const;

 private:
  const HloModule* module_;
  const HloAliasAnalysis* hlo_alias_analysis_;
  absl::flat_hash_map<const HloComputation*, MemoryPressureState>
      memory_pressure_states_;
  mutable absl::Mutex pressure_metadata_cache_mu_;
  mutable absl::flat_hash_map<const HloComputation*,
                              std::unique_ptr<MemoryPressureMetadata>>
      pressure_metadata_cache_ ABSL_GUARDED_BY(pressure_metadata_cache_mu_);
  BufferInfoTracker buffer_tracker_;
  int64_t memory_peak_ = 0;
  bool top_down_scheduling_ = false;
};

// Light data structure containing information on the schedule of a computation,
// can be used by a heuristic to evaluate the quality of the schedule.
struct ComputationScheduleInfo {
  double total_wasted_cycles;
  uint64_t peak_memory;
};

// Comparator for the ready set. This class represents the priority policies
// for the nodes in the ready set. The policy can be whatever is appropriate to
// reduce the execution time of the graph or achieve interesting properties
// (best CMEM/VMEM allocations, latency hiding, memory pressure ... etc).
class ReadySetLt;

// Implementation of the default scheduling algorithm.
class DefaultSchedulerCore : public SchedulerCore {
  friend class ReadySetLt;

 public:
  using ReadyQueueSet = std::vector<HloGraphNode*>;
  using ResourceMap = absl::flat_hash_map<int64_t, int64_t>;
  using ShouldSkipNodeFunction = std::function<bool(const HloGraphNode*)>;

  struct SchedulingState;

  // Class used to cache expensive information. Currently memory pressure
  // changes are cached. The caching is invalidated at the end of the scheduling
  // process for this next candidate. The information shouldn't survive across
  // scheduling two different instructions.
  struct ScheduleCandidate {
    void set_pressure_change(std::pair<const int64_t, const int64_t> v) {
      pressure_change_first = v.first;
      pressure_change_second = v.second;
      has_pressure_change = true;
    }
    void set_estimated_connected_send_ready_time(HloGraphNode::TimeCost v) {
      estimated_connected_send_ready_time = v;
      has_estimated_connected_send_ready_time = true;
    }
    void set_resource_constrained(bool v) {
      resource_constrained = v;
      has_resource_constrained = true;
    }

    HloGraphNode* node = nullptr;
    const SchedulingState* scheduling_state = nullptr;

    // Fields below are valid if the corresponding has_... field is true

    bool has_pressure_change = false;
    bool has_estimated_connected_send_ready_time = false;
    bool has_resource_constrained = false;

    int64_t pressure_change_first;
    int64_t pressure_change_second;
    HloGraphNode::TimeCost estimated_connected_send_ready_time;
    bool resource_constrained;
  };

  struct CandidateResult {
    const ScheduleCandidate& result;
    const char* reason;
  };

  using TargetSchedulingRule = std::function<std::optional<CandidateResult>(
      ScheduleCandidate&, ScheduleCandidate&)>;

  // Returns nullopt if both conditions are equal, otherwise returns the
  // candidate corresponding to the true condition.
  static inline std::optional<CandidateResult> ChooseBestCandidate(
      bool first_cond, const ScheduleCandidate& first_candidate,
      bool second_cond, const ScheduleCandidate& second_candidate,
      const char* reason) {
    if (first_cond == second_cond) {
      return std::nullopt;
    }
    return CandidateResult{first_cond ? first_candidate : second_candidate,
                           reason};
  }

  absl::Status SetGraphProcessingHook(
      const SchedulerCore::GraphProcessingHook& hook) override {
    default_graph_processing_hook_ = hook;
    return absl::OkStatus();
  }

  // The scheduling state contains everything that is required for the
  // bookkeeping of the scheduling algorithm. Functions that perform operations
  // over the scheduling state can directly operate on the state contained into
  // this struct instead of having to pass many individual pointers to elements
  // of the state.
  struct SchedulingState : public SchedulerCore::SchedulingState {
    std::unique_ptr<HloScheduleGraph> sched_graph;
    // Ready set for the nodes. Its ordered by our heuristic defined in
    // ReadySetLt.
    ReadyQueueSet ready_set;
    // Maximum allowed number of overlapping instructions using the key resource
    // type.
    ResourceMap max_concurrent_resource;
    // New scheduling sequence produced by the scheduler. This is in reversed
    // order (because we schedule bottom up). This will be required to be
    // reversed before assigning to the HloSchedule.
    std::vector<HloInstruction*> new_sequence_reversed;

    // Memory pressure during and after an instruction in a schedule.
    // (memory_after, memory_peak)
    absl::flat_hash_map<const HloInstruction*, std::pair<int64_t, int64_t>>
        memory_trace;
    // Units of time passed in the schedule. To keep track of latency hiding.
    HloGraphNode::TimeCost current_time = 0;
    // Resources and corresponding occupiers in flight.
    absl::flat_hash_map<int64_t, absl::flat_hash_set<const HloInstruction*>>
        resource_occupiers_in_flight;
    // Number of instructions using the key resource type in the set waiting to
    // be scheduled.
    ResourceMap resource_users_in_queue;
    // Number of nodes scheduled.
    int64_t scheduled_count = 0;
    // Class returning information about instruction cost and latency between
    // instructions.
    const LatencyEstimator* latency_estimator;
    // Class used to track which instructions are async instructions and which
    // async instructions computations contain. It also tracks target defined
    // states related to the async instructions.
    const AsyncTracker* async_tracker;
    // Tracker of memory pressure for the computation.
    const MemoryPressureMetadata* pressure_metadata;
    MemoryPressureTracker memory_pressure_tracker;
    // Vector containing a list of nodes that aren't ready to schedule yet in
    // order of time when they are going to become ready.
    std::vector<const HloGraphNode*> next_ready_stack;
    // List of the graph edges currently occupying the key shareable resource
    // with projected finish times.
    absl::flat_hash_map<
        int64_t, std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>>
        shareable_resource_occupiers;
    // List of the graph nodes that release selective resources.
    std::vector<HloGraphNode*> selective_resource_releasers;
    // Similar to ready set, but only contains the no-op instructions.
    ReadyQueueSet nop_set;
    // Number of {scheduled, all} nodes that are a successor for the given
    // annotation.
    struct NumSuccessorsForAnnotation {
      int64_t scheduled = 0;
      int64_t all = 0;
    };
    struct NumPredecessorsForAnnotation {
      int64_t scheduled = 0;
      int64_t all = 0;
    };
    absl::flat_hash_map<int64_t, NumSuccessorsForAnnotation>
        num_successors_for_annotation;
    absl::flat_hash_map<int64_t, NumPredecessorsForAnnotation>
        num_predecessors_for_annotation;
    // List of annotations that are ready to be scheduled.
    absl::InlinedVector<int64_t, 2> ready_annotations;
    // List of annotated nodes that are ready to be scheduled.
    ReadyQueueSet annotation_ready;
    // Annotation that is currently being scheduled.
    int64_t ongoing_annotation = kInvalidAnnotation;
    // If this set is not empty it means that we shouldn't schedule any more
    // annotated nodes until empty.
    absl::flat_hash_set<HloGraphNode*> nodes_holding_annotations;
    // Reference to this scheduler run configuration.
    const SchedulerConfig& config;
    void Reset() override;

    SchedulingState(
        const HloInstructionSequence* instr_sequence,
        std::shared_ptr<const SchedulingContext>& scheduling_context,
        const MemoryPressureMetadata* pressure_metadata,
        const SchedulerConfig& config, std::unique_ptr<HloScheduleGraph> graph)
        : sched_graph(std::move(graph)),
          latency_estimator(scheduling_context->GetLatencyEstimator().get()),
          async_tracker(scheduling_context->GetAsyncTracker().get()),
          pressure_metadata(pressure_metadata),
          memory_pressure_tracker(pressure_metadata),
          config(config) {}
  };

  using OverlapLimitRule =
      std::function<bool(const SchedulingState&, const HloGraphNode*)>;
  using PostProcessingFn = std::function<void(SchedulingState&)>;

  DefaultSchedulerCore(
      std::shared_ptr<const SchedulingContext> scheduling_context,
      const SchedulerConfig& config,
      TargetSchedulingRule target_scheduling_rule = nullptr,
      TargetSchedulingRule early_target_scheduling_rule = nullptr,
      PostProcessingFn post_processing_fn = nullptr,
      OverlapLimitRule scheduling_instruction_crosses_overlap_limit = nullptr)
      : config_(config),
        target_scheduling_rule_(target_scheduling_rule),
        early_target_scheduling_rule_(early_target_scheduling_rule),
        post_processing_fn_(post_processing_fn),
        scheduling_instruction_crosses_overlap_limit_(
            scheduling_instruction_crosses_overlap_limit),
        scheduling_context_(std::move(scheduling_context)),
        top_down_scheduling_(config.top_down_scheduling) {}

  absl::Status InitializeScheduler(const HloModule* module) override;

  absl::Status CaptureScheduleProto() override {
    absl::MutexLock lock(&schedule_proto_mu_);
    schedule_proto_ = ScheduleProto();
    *schedule_proto_->mutable_hlo_module() = module_->ToProto();

    return absl::OkStatus();
  }

  absl::StatusOr<ScheduleProto> GetCapturedScheduleProto() override {
    absl::MutexLock lock(&schedule_proto_mu_);
    if (!schedule_proto_.has_value()) {
      return absl::FailedPreconditionError("Schedule proto not captured.");
    }
    return schedule_proto_.value();
  }

  absl::StatusOr<std::shared_ptr<SchedulerCore::SchedulingState>>
  MakeSchedulingState(const HloComputation* computation) override;

  virtual std::unique_ptr<ReadySetLt> CreateReadySetComparator(
      SchedulingState& sched_state) const;

  absl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      const HloComputation* computation) override;
  absl::StatusOr<std::vector<HloInstruction*>> ScheduleComputation(
      const HloComputation* computation,
      std::shared_ptr<SchedulerCore::SchedulingState> sched_state) override;
  std::shared_ptr<SchedulerCore::SchedulingState> GetSchedulingState() override;
  static bool AddOccupierToResource(
      HloGraphNode::TimeCost current_time, HloEdge& new_edge,
      std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>& occupiers,
      bool top_down_scheduling = false);
  static bool DeleteOccupierFromResource(
      HloGraphNode::TimeCost current_time, HloEdge& edge,
      std::vector<std::pair<HloEdge*, HloGraphNode::TimeCost>>& occupiers);
  static bool DefaultSchedulingInstructionCrossesOverlapLimit(
      const SchedulingState& sched_state, const HloGraphNode* node);
  int64_t GetMemoryPeak() override {
    return module_pressure_state_->GetMemoryPeak();
  }
  int64_t GetMemoryPeakForComputation(const HloComputation* computation) const {
    return module_pressure_state_->GetPressureStateForComputation(computation)
        .memory_peak;
  }

  uint64_t GetMemoryLimit() override { return config_.memory_limit; }
  void SetMemoryLimit(uint64_t new_limit) override {
    this->config_.memory_limit = new_limit;
  }
  int64_t GetRerunTimes() override { return config_.rerun; }
  const SchedulerConfig& GetConfig() const { return config_; }

  // Returns the amount of resources an annotation group needs. The amount of
  // resources needed is schedule-order dependent. This function returns the
  // minimum or the maximum amount of resources needed for the given annotation
  // group based on the value of get_max_resources.
  absl::flat_hash_map<int64_t, int64_t> GetNumResourcesNeededForAnnotation(
      const SchedulingState& sched_state, int64_t annotation,
      bool get_max_resources = false);

  // Returns true if the given annotation group crosses the overlap limit.
  // If use_max_resources is true, the maximum amount of resources needed for
  // the annotation group is used to compare against the overlap limit.
  // Otherwise, the minimum amount of resources needed for the annotation group
  // is used.
  bool SchedulingAnnotationCrossesOverlapLimit(
      const SchedulingState& sched_state, int64_t annotation,
      bool use_max_resources = false);

  int64_t GetNumPredecessorsForAnnotation(const SchedulingState& sched_state,
                                          int64_t annotation) const;

  int64_t GetNumSuccessorsForAnnotation(const SchedulingState& sched_state,
                                        int64_t annotation) const;

  // Tries to schedule any of the ready annotation groups using either the
  // maximum or minimum amount of resources needed for the annotation group
  // based on value of use_max_resources. Returns true if any annotation group
  // is scheduled, false otherwise.
  absl::StatusOr<bool> TryScheduleOneAnnotationGroup(
      DefaultSchedulerCore::SchedulingState* sched_state,
      const HloComputation* computation, bool use_max_resources);

  ScheduleProto::ComputationScheduleProto ComputationScheduleToProto(
      const HloComputation* computation, const SchedulingState& sched_state,
      const LatencyEstimator& estimator,
      const std::vector<HloInstruction*>& instructions);

  ModulePressureState* GetModulePressureState() {
    return module_pressure_state_.get();
  }

 protected:
  virtual std::unique_ptr<HloScheduleGraph> CreateScheduleGraph(
      const std::vector<HloInstruction*>* instructions,
      std::shared_ptr<const SchedulingContext> context,
      std::shared_ptr<const HloReachabilityMap> reachability = nullptr) const;
  virtual void LogInstruction(const HloInstruction* instr) const;

  // Schedules the given annotated node.
  absl::Status AnnotatedSchedulingStep(
      HloGraphNode* node,
      DefaultSchedulerCore::SchedulingState* sched_state) const;
  // Schedules all of the nodes with the given annotation in computation.
  absl::Status ScheduleAnnotation(
      const HloComputation* computation, int64_t annotation,
      DefaultSchedulerCore::SchedulingState* sched_state) const;
  // Update node that has been scheduled.
  virtual absl::StatusOr<HloGraphNode::TimeCost> ScheduleNode(
      HloGraphNode* n, SchedulingState* sched_state) const;
  // Perform the scheduling of one or more instructions. Called every time the
  // ready set is not empty.
  virtual absl::Status SchedulingStep(SchedulingState* sched_state);
  // Pick a node to schedule according to cost model.
  virtual absl::StatusOr<HloGraphNode*> FindAndExtractBestNodeAvailable(
      SchedulingState& sched_state,
      DefaultSchedulerCore::ShouldSkipNodeFunction should_skip_node);

  std::unique_ptr<ModulePressureState> module_pressure_state_;
  absl::flat_hash_map<const HloComputation*,
                      std::unique_ptr<MemoryPressureMetadata>>
      pressure_metadata_;
  SchedulerConfig config_;
  TargetSchedulingRule target_scheduling_rule_ = nullptr;
  TargetSchedulingRule early_target_scheduling_rule_ = nullptr;
  PostProcessingFn post_processing_fn_ = nullptr;
  OverlapLimitRule scheduling_instruction_crosses_overlap_limit_ = nullptr;
  bool is_default_scheduling_instruction_crosses_overlap_limit_ = false;
  std::unique_ptr<AnnotationTracker> annotation_tracker_;
  mutable absl::Mutex schedule_proto_mu_;
  std::optional<ScheduleProto> schedule_proto_
      ABSL_GUARDED_BY(schedule_proto_mu_);
  const HloModule* module_ = nullptr;
  SchedulerCore::GraphProcessingHook default_graph_processing_hook_;
  std::shared_ptr<const SchedulingContext> scheduling_context_;
  mutable absl::Mutex latest_sched_state_mu_;
  std::shared_ptr<SchedulerCore::SchedulingState> latest_sched_state_
      ABSL_GUARDED_BY(latest_sched_state_mu_);
  std::shared_ptr<const HloReachabilityMap> GetReachabilityMap(
      const HloComputation* computation);

  mutable absl::Mutex reachability_cache_mu_;
  absl::flat_hash_map<const HloComputation*,
                      std::shared_ptr<const HloReachabilityMap>>
      reachability_cache_ ABSL_GUARDED_BY(reachability_cache_mu_);
  bool top_down_scheduling_ = false;
};

class ReadySetLt {
 public:
  using SchedulingState = DefaultSchedulerCore::SchedulingState;
  using ScheduleCandidate = DefaultSchedulerCore::ScheduleCandidate;
  using TargetSchedulingRule = DefaultSchedulerCore::TargetSchedulingRule;
  using CandidateResult = DefaultSchedulerCore::CandidateResult;

  explicit ReadySetLt(DefaultSchedulerCore::SchedulingState& sched_state,
                      const DefaultSchedulerCore* core)
      : sched_state_(sched_state), core_(core) {}
  virtual ~ReadySetLt() = default;

  virtual bool MaybeUpdate(DefaultSchedulerCore::ScheduleCandidate& a,
                           DefaultSchedulerCore::ScheduleCandidate& b,
                           const char** reason) const;

  virtual bool AIsBetterThanB(DefaultSchedulerCore::ScheduleCandidate& a,
                              DefaultSchedulerCore::ScheduleCandidate& b,
                              const char** reason) const;

 protected:
  template <typename T>
  static int ThreeWay(T avalue, T bvalue) {
    if (ABSL_PREDICT_TRUE(avalue == bvalue)) {
      return 0;
    }
    return (avalue < bvalue) ? -1 : 1;
  }

  template <typename T>
  static std::optional<bool> CmpExplicit(T pa, T pb, const char* reason_str,
                                         const char** reason) {
    if (int v = ThreeWay(pa, pb)) {
      *reason = reason_str;
      return v > 0;
    }
    return std::nullopt;
  }

  template <typename T>
  static std::optional<bool> CmpDirectional(bool top_down, T pa, T pb,
                                            const char* reason_str,
                                            const char** reason) {
    return top_down ? CmpExplicit(pa, pb, reason_str, reason)
                    : CmpExplicit(pb, pa, reason_str, reason);
  }
  std::optional<bool> MemoryPressurePolicy(
      const DefaultSchedulerCore::SchedulingState& state,
      const HloGraphNode* an, std::pair<int64_t, int64_t>& a_increase,
      const HloGraphNode* bn, std::pair<int64_t, int64_t>& b_increase,
      const char** reason) const;

  std::optional<bool> ReleaseStartPolicy(
      const DefaultSchedulerCore::SchedulingState& state,
      const HloGraphNode* an, const HloGraphNode* bn,
      const char** reason) const;

  bool AsyncDepth0CandidateCondition(
      const DefaultSchedulerCore::SchedulingState& state,
      DefaultSchedulerCore::ScheduleCandidate& a,
      const HloGraphNode* a_node) const;

  bool ShouldScheduleAsyncDone(
      const DefaultSchedulerCore::SchedulingState& state,
      DefaultSchedulerCore::ScheduleCandidate& gn_cand,
      const HloGraphNode* gn_node) const;

  bool ShouldDelaySendHostDone(
      const DefaultSchedulerCore::SchedulingState& state,
      DefaultSchedulerCore::ScheduleCandidate& gn_cand,
      const HloGraphNode* gn_node) const;

  bool ShouldScheduleAsyncStart(
      const DefaultSchedulerCore::SchedulingState& state,
      DefaultSchedulerCore::ScheduleCandidate& gn_cand,
      const HloGraphNode* gn_node) const;

  std::optional<bool> IsValuableForSelectiveOverlap(
      const DefaultSchedulerCore::SchedulingState& state,
      DefaultSchedulerCore::ScheduleCandidate& a,
      DefaultSchedulerCore::ScheduleCandidate& b, const char** reason) const;

  static std::optional<bool> InvokeTargetSchedulingFunction(
      DefaultSchedulerCore::TargetSchedulingRule func,
      DefaultSchedulerCore::ScheduleCandidate& a,
      DefaultSchedulerCore::ScheduleCandidate& b, const char** reason);

  void UpdateCandidateResourceConstrained(
      const DefaultSchedulerCore::SchedulingState& state,
      DefaultSchedulerCore::ScheduleCandidate& cand,
      const HloGraphNode* cand_node) const;

  bool IsResourceConstrained(const DefaultSchedulerCore::SchedulingState& state,
                             DefaultSchedulerCore::ScheduleCandidate& cand,
                             const HloGraphNode* cand_node) const;

  HloGraphNode::TimeCost PastDueCyclesForNonextendableResource(
      const DefaultSchedulerCore::SchedulingState& state,
      const HloGraphNode* cand_node) const;

  std::pair<int64_t, int64_t> GetMemoryPressureChanges(
      const DefaultSchedulerCore::SchedulingState& state,
      DefaultSchedulerCore::ScheduleCandidate& cand,
      const HloGraphNode* cand_node) const;

  int64_t GetNumConflictingSerialResources(
      const DefaultSchedulerCore::SchedulingState& state,
      DefaultSchedulerCore::ScheduleCandidate& cand,
      const HloGraphNode* cand_node) const;

  DefaultSchedulerCore::SchedulingState& sched_state_;
  const DefaultSchedulerCore* core_;
};

// A scheduler oriented to hiding latencies of operations that can run in
// parallel with other operations.
class LatencyHidingScheduler : public HloModulePass {
 public:
  struct SchedulerStatistics {
    const HloComputation* computation = nullptr;
    double all_gather_wasted_cycles = 0;
    double all_reduce_wasted_cycles = 0;
    double collective_broadcast_wasted_cycles = 0;
    double collective_permute_wasted_cycles = 0;
    double all_to_all_wasted_cycles = 0;
    double ragged_all_to_all_wasted_cycles = 0;
    double reduce_scatter_wasted_cycles = 0;
    double send_wasted_cycles = 0;
    double recv_wasted_cycles = 0;
    double call_wasted_cycles = 0;
    double total_cycles = 0;
    int64_t memory_pressure_peak = 0;

    double GetTotalWastedCycles() const {
      return all_gather_wasted_cycles + all_reduce_wasted_cycles +
             collective_broadcast_wasted_cycles +
             collective_permute_wasted_cycles + all_to_all_wasted_cycles +
             ragged_all_to_all_wasted_cycles + reduce_scatter_wasted_cycles +
             send_wasted_cycles + recv_wasted_cycles + call_wasted_cycles;
    }

    ScheduleProto::SchedulerStatisticsProto ToProto() const;
    std::string ToString() const;
  };

  LatencyHidingScheduler(
      std::shared_ptr<const SchedulingContext> scheduling_context,
      std::shared_ptr<SchedulerCore> scheduler_core)
      : scheduling_context_(std::move(scheduling_context)),
        scheduler_core_(std::move(scheduler_core)) {}
  constexpr static absl::string_view kName = "latency-hiding-scheduler";
  absl::string_view name() const override { return kName; }

  std::shared_ptr<const SchedulingContext> scheduling_context() const {
    return scheduling_context_;
  }

  // Returns some printable statistics about the latency hiding for
  // operations that can run in parallel to help evaluating the performance of
  // the scheduler and improve it.
  // Optionally the caller can pass in the alias analysis and module pressure
  // state to save the time to construct them within the function. This is
  // useful when we repeatedly call this function across computations within the
  // same module.
  static SchedulerStatistics LatencyHidingStatistics(
      const HloComputation* computation,
      absl::Span<const HloInstruction* const> candidate_sequence,
      std::shared_ptr<const SchedulingContext> scheduling_context,
      const ModulePressureState* pressure_state = nullptr,
      MemoryPressureTracker* memory_pressure_tracker = nullptr,
      std::shared_ptr<SchedulerCore::SchedulingState> sched_state = nullptr);

  virtual void LogScheduleStatistics(const HloComputation* computation);

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 protected:
  std::shared_ptr<const SchedulingContext> scheduling_context_;
  std::shared_ptr<SchedulerCore> scheduler_core_;
  std::vector<HloComputation*> computations_to_schedule_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_HLO_SCHEDULING_LATENCY_HIDING_SCHEDULER_H_
