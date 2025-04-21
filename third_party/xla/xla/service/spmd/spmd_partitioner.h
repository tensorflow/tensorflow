/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_SPMD_PARTITIONER_H_
#define XLA_SERVICE_SPMD_SPMD_PARTITIONER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/literal.h"
#include "xla/service/call_graph.h"
#include "xla/service/custom_call_sharding_helper.h"
#include "xla/service/dot_as_convolution_util.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace spmd {

// Enum representing the partitioning methods for gather and scatter.
enum class GatherScatterPartitioningMethod {
  kExplicitBatch,
  kIndexParallel,
  kOperandPassthrough,
  kTrivialSlicedOperand,
  kIndexPassthrough,
};

struct SpmdPartitionerOptions {
  // Always exchange halo on LHS for all convolutions. If false, backprop filter
  // convolution exchanges halo on RHS.
  bool conv_halo_exchange_always_on_lhs = true;

  // The number of instructions to be reported for the highest memory profile
  // instructions.
  int64_t report_instruction_count = 5;

  // The minimum size in MiB of an einsum operand to be considered using
  // windowed implementation in an HLO loop.
  int64_t threshold_for_windowed_einsum_mib = 256;

  // Whether unroll windowed einsum loop by degree of two.
  bool unroll_windowed_einsum = false;

  // Whether doing bidirectional collective permute in windowed einsum loop.
  bool bidirectional_windowed_einsum = false;

  // Whether the entry computations' signature could change after partitioning.
  bool allow_module_signature_change = false;

  // Whether to use cached all-gather to avoid repeatedly replicate a tiled
  // tensor. If it is set to false, the result tends to be more
  // memory-efficient, and the compiler can use the ScheduleAwareAllGatherCSE
  // pass to CSE some all-gathers which are relatively close to each other.
  bool cache_all_gather = true;

  // When making a compromise between windowed einsum speed and memory usage
  // prefer the former if true.
  bool choose_faster_windowed_einsum_over_mem = false;

  // Whether doing bidirectional communication when decomposing independent
  // all-gathers.
  bool bidirectional_decomposed_all_gather = false;

  // Whether to skip checking the numbers and shardings of windowed einsum's
  // users.
  bool skip_checking_windowed_einsum_users = false;

  // Enables windowed einsum for operand all-gather.
  bool enable_windowed_einsum_for_all_gather = true;
  // Enables windowed einsum for result reduce-scatter.
  bool enable_windowed_einsum_for_reduce_scatter = true;

  // Whether disable rewrite for dots that share the same
  // operand as an already rewritten windowed einsum loop.
  bool disable_ag_rewrite_for_multiple_consumers = false;

  // Partitioning method to prioritize for gather operations.
  std::vector<GatherScatterPartitioningMethod>
      preferred_gather_partition_methods = {
          GatherScatterPartitioningMethod::kExplicitBatch,
          GatherScatterPartitioningMethod::kIndexParallel};

  // Partitioning method to prioritize for scatter operations.
  std::vector<GatherScatterPartitioningMethod>
      preferred_scatter_partition_methods = {
          GatherScatterPartitioningMethod::kExplicitBatch,
          GatherScatterPartitioningMethod::kIndexParallel};

  // The minimum size to enable windowed einsum in total bytes.
  // This combines sizes in bytes of both operands.
  // When it's set, it will override threshold_for_windowed_einsum_mib.
  std::optional<int64_t> total_bytes_windowed_einsum_threshold = std::nullopt;
};

// Class to wrap the computation builder to capture information during SPMD
// transformation.
class SpmdBuilder : public HloComputation::Builder {
 public:
  SpmdBuilder(const std::string& name, HloInstruction* hlo)
      : HloComputation::Builder(name) {
    visiting_hlo_ = hlo;
  }

  HloInstruction* AddInstruction(
      std::unique_ptr<HloInstruction> instruction) override;

  const std::vector<HloInstruction*>& derived_instructions(
      HloInstruction* hlo) {
    return instructions_.at(hlo);
  }

  void set_visiting_hlo(HloInstruction* hlo) {
    visiting_hlo_ = hlo;
    instructions_[hlo];
  }

  HloInstruction* visiting_hlo() const { return visiting_hlo_; }

  // Wrapper of queries to broadcast_dims_.
  std::optional<const absl::flat_hash_set<int64_t>*> BroadcastDimsForCreatedHlo(
      const HloInstruction* hlo) {
    auto it = broadcast_dims_.find(hlo);
    if (it == broadcast_dims_.end()) {
      return std::nullopt;
    }
    return &it->second;
  }

 private:
  // Sets the broadcast dims for the newly added/created hlo.
  void SetBroadcastDimsForAddedHlo(const HloInstruction& hlo);

  void SetBroadcastDimsForReshape(const HloInstruction& hlo);

  void SetBroadcastDimsForTranspose(const HloInstruction& hlo);

  void SetBroadcastDimsForPad(const HloInstruction& hlo);

  void SetBroadcastDimsForSlice(const HloInstruction& hlo);

  void SetBroadcastDimsForElementwise(const HloInstruction& hlo);
  // Currently visiting instruction.
  HloInstruction* visiting_hlo_;

  // Map from the currently visiting (old) instruction to new instructions
  // created during SPMD partitioning.
  HloInstructionMap<std::vector<HloInstruction*>> instructions_;

  // Maps from each created instruction to a set of dimensions that are from
  // broadcasts or elementwise ops over broadcasts. This means elements along
  // these dimensions have the same value.
  absl::flat_hash_map<const HloInstruction*, absl::flat_hash_set<int64_t>>
      broadcast_dims_;
};

// A set of functions that create the cross-partition collective ops.
struct SPMDCollectiveOpsCreator {
  // Function used to create a partition ID HLO.
  std::function<HloInstruction*(SpmdBuilder*)> create_partition_id;

  // Function used to create a cross-partition all-reduce HLO.
  std::function<HloInstruction*(
      SpmdBuilder*, HloInstruction* operand, HloComputation* reduction,
      const std::vector<std::vector<int64_t>>& partition_subgroups,
      int64_t channel_id)>
      create_cross_partition_all_reduce;

  // Function used to create a cross-partition all-reduce HLO using device list
  // in iota format. This function is optional: if it is a nullptr, use
  // create_cross_partition_all_reduce.
  // TODO(b/316622399): Merge this and create_cross_partition_all_reduce into a
  // function that uses CollectiveDeviceList.
  std::function<HloInstruction*(
      SpmdBuilder*, HloInstruction* operand, HloComputation* reduction,
      const IotaReplicaGroupList& partition_group_list, int64_t channel_id)>
      create_cross_partition_all_reduce_with_iota_device_list;

  // Function used to create a cross-partition collective-permute HLO.
  std::function<HloInstruction*(
      SpmdBuilder*, HloInstruction* operand,
      std::vector<std::pair<int64_t, int64_t>>& src_dst_pairs,
      int64_t next_channel_id)>
      create_cross_partition_collective_permute;

  // Function used to create a cross-partition all-to-all HLO.
  std::function<HloInstruction*(
      SpmdBuilder*, absl::Span<HloInstruction* const> operands,
      const std::vector<std::vector<int64_t>>& partition_subgroups,
      int64_t channel_id, std::optional<int64_t> split_dimension)>
      create_cross_partition_all_to_all;

  // Function used to create a cross-partition all-gather HLO. This is optional:
  // if it is nullptr, the partitioner will use all-reduce instead.
  std::function<HloInstruction*(
      SpmdBuilder*, HloInstruction* operand, const Shape& ag_shape,
      const std::vector<std::vector<int64_t>>& partition_subgroups,
      int64_t channel_id, int64_t all_gather_dimension)>
      create_cross_partition_all_gather;

  // Function used to create a cross-partition all-gather HLO using device list
  // in iota format. This function is optional: if it is a nullptr, use
  // create_cross_partition_all_gather.
  // TODO(b/316622399): Merge this and create_cross_partition_all_gather into a
  // function that uses CollectiveDeviceList.
  std::function<HloInstruction*(
      SpmdBuilder*, HloInstruction* operand, const Shape& ag_shape,
      const IotaReplicaGroupList& partition_group_list, int64_t channel_id,
      int64_t all_gather_dimension)>
      create_cross_partition_all_gather_with_iota_device_list;
};

// Create a default SPMDCollectiveOpsCreator.
SPMDCollectiveOpsCreator GetDefaultCollectiveOpsCreator(int64_t num_partitions,
                                                        int64_t num_replicas);

// Logger to report memory usage during SPMD partitioning.
class SpmdLogger {
 public:
  SpmdLogger(int64_t report_instruction_count, bool disabled)
      : report_instruction_count_(report_instruction_count),
        disabled_(disabled) {}
  static std::string ReportBeforePartition(const HloModule& module,
                                           int64_t report_instruction_count);
  static std::string ReportAfterPartition(const HloModule& module,
                                          int64_t report_instruction_count);

  // Registers the logging for the groups of instructions created to transform
  // the given hlo.
  void RegisterLogEntry(HloInstruction* hlo,
                        const std::vector<HloInstruction*>& group);

  std::string MakeReport();

 private:
  template <typename F>
  static std::string ReportMemoryUsage(const HloModule& module, const F& filter,
                                       int64_t report_instruction_count);

  // A vector of logging messages (one for each original HLO instruction), where
  // the first integer of the pair represents the size of the HBM used.
  std::vector<std::pair<int64_t, std::string>> entries_;

  int64_t report_instruction_count_;

  // Note that we allow creating a *disabled* logger when logging is not
  // enabled, in which case it is supposed to avoid doing any potentially
  // expensive work. The logger is still created in this case and passed to the
  // users to help avoid changing current call sites.
  const bool disabled_;
};

class SpmdPartitioningVisitor;

class SpmdPartitioner : public HloModulePass {
 public:
  SpmdPartitioner(int64_t num_partitions, int64_t num_replicas,
                  SpmdPartitionerOptions options);
  SpmdPartitioner(int64_t num_partitions, int64_t num_replicas,
                  SpmdPartitionerOptions options,
                  SPMDCollectiveOpsCreator collective_ops_creator)
      : num_partitions_(num_partitions),
        num_replicas_(num_replicas),
        options_(std::move(options)),
        collective_ops_creator_(std::move(collective_ops_creator)) {}
  absl::string_view name() const override { return "spmd-partitioning"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Transforms the given computation with SPMD instructions, replacing it with
  // a new computation.
  absl::StatusOr<bool> PartitionComputation(HloComputation* computation,
                                            const HloSharding& root_sharding,
                                            int64_t* next_channel_id,
                                            SpmdLogger* logger,
                                            const CallGraph& call_graph);

  // Creates all-gather(s) based on HloSharding. Can be overridden to customize.
  // The default uses a single all-gather even if there are multiple sharded
  // dimensions, and adds potential reshapes and transposes to achieve that.
  // If it returns false, the partitioner will fall back to all-reduce.
  // `selected_dims` specifies the dimensions along which the all-gather happens
  // in the tiled sharding, which allows potentially creating a subgroup
  // all-gather.
  virtual HloInstruction* AllGatherShards(
      SpmdBuilder* b, HloInstruction* operand, const HloSharding& sharding,
      int64_t* next_channel_id, absl::Span<const int64_t> selected_dims,
      const SPMDCollectiveOpsCreator& collectives_creator);

  // Creates all-reduce(s) across devices along selected_dims in sharding. Can
  // be overridden to customize.
  virtual HloInstruction* AllReduceAlongShardingDims(
      SpmdBuilder* b, HloInstruction* operand, const HloSharding& sharding,
      int64_t* next_channel_id, absl::Span<const int64_t> selected_dims,
      const SPMDCollectiveOpsCreator& collectives_creator,
      HloComputation* reduction);

  const SpmdPartitionerOptions& options() { return options_; }

  virtual std::unique_ptr<SpmdPartitioningVisitor> CreateVisitor(
      HloComputation* computation, int64_t num_partitions, int64_t num_replicas,
      const SPMDCollectiveOpsCreator& collective_ops_creator,
      int64_t* next_channel_id, SpmdLogger* logger,
      SpmdPartitionerOptions options, const CallGraph& call_graph);

  // Estimate the memory cost for an op, override this for target-specific
  // op buffer implementation.
  virtual int64_t MemoryCostInBytes(HloInstruction* hlo);

  // Estimate the communication cost for a collective op, override this for
  // target-specific collective implementation.
  virtual int64_t CommunicationCostInBytes(HloInstruction* hlo);

  const absl::flat_hash_set<absl::string_view>& execution_threads() const {
    return execution_threads_;
  }

  // Update module's parameter and output sharding information, based on the
  // sharding information of the module's parameters and outptuts.
  static void RecordInputsOutputsSharding(HloModule* module);

 protected:
  // This is the internal implementation for AllGatherShards(), returns a pair
  // of hlo instructions whose first element is the result of the all-gather
  // shard(which might not be the all-gather itself and it could go through
  // some other formatting instructions), and the second element is the
  // all-gather being generated or nullptr is no all-gather is generated.
  std::pair<HloInstruction*, HloInstruction*> AllGatherShardsInternal(
      SpmdBuilder* b, HloInstruction* operand, const HloSharding& sharding,
      int64_t* next_channel_id, absl::Span<const int64_t> selected_dims,
      const SPMDCollectiveOpsCreator& collectives_creator, bool per_dim_ag);

  HloInstruction* AllReduceAlongShardingDimsInternal(
      SpmdBuilder* b, HloInstruction* operand, const HloSharding& sharding,
      int64_t* next_channel_id, absl::Span<const int64_t> selected_dims,
      const SPMDCollectiveOpsCreator& collectives_creator,
      HloComputation* reduction, bool per_dim_ar);

  // Verifies that the sharding of instructions in the module are valid, and
  // also fill in missing sharding information.
  virtual absl::Status PreprocessSharding(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Returns if the given side-effecting instruction is allowed to have
  // replicated sharding.
  virtual bool CanSideEffectingHaveReplicatedSharding(
      const HloInstruction* hlo) {
    if (hlo->opcode() == HloOpcode::kCustomCall) {
      if (auto* partitioner =
              GetCustomCallPartitioner(hlo->custom_call_target())) {
        return partitioner->CanSideEffectingHaveReplicatedSharding();
      }
    }
    return hlo->opcode() == HloOpcode::kInfeed ||
           hlo->opcode() == HloOpcode::kOutfeed;
  }

  // Preprocesses the graph to simplify some communication patterns. E.g., merge
  // pad->slice into a single pad with potentially negative padding to avoid
  // multiple halo exchanges.
  absl::Status PreprocessHlos(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);

  void set_execution_threads(
      const absl::flat_hash_set<absl::string_view>& execution_threads) {
    execution_threads_ = execution_threads;
  }

  const int64_t num_partitions_;
  const int64_t num_replicas_;

  SpmdPartitionerOptions options_;
  SPMDCollectiveOpsCreator collective_ops_creator_;
  std::vector<std::vector<int64_t>> device_groups_;
  absl::flat_hash_set<absl::string_view> execution_threads_;
};

// Class describes partition state of the data represented by an HLO created
// during SPMD partitioning pass.
//
// Data on some devices may include padding region, if the base (full) shape
// could not be evenly partitioned.
class PartitionedHlo {
 public:
  // Return value for ReshardAsWindowedInput which describes the resharded HLO,
  // the window for the user on the shard, and if necessary, the dynamic slice
  // offsets to be applied to the output of the op being sharded.
  struct WindowedInputShardReturnValue {
    HloInstruction* sharded_input;
    Window shard_window;
    std::optional<std::vector<HloInstruction*>> dynamic_slice_index_on_output;
  };
  // A cache for resharding each partitioned HLO.
  struct ReshardCache {
    struct PerHloCache {
      absl::flat_hash_map<HloSharding, PartitionedHlo> reshard_cache;
      std::vector<
          std::tuple<HloSharding, Window, WindowedInputShardReturnValue>>
          window_reshard_cache;
    };
    // Use absl::node_hash_map for pointer stability.
    absl::node_hash_map<HloInstruction*, PerHloCache> per_hlo_cache;
    // Caches for nested partitioning of grouped sharding. Each string key
    // represents a unique way of grouping devices.
    absl::flat_hash_map<std::string, std::unique_ptr<ReshardCache>>
        groupd_caches;
  };
  struct PartitioningState {
    SpmdBuilder* b;
    HloModule* module;
    int64_t num_replicas;
    HloInstruction* partition_id;
    SPMDCollectiveOpsCreator collective_ops_creator;
    int64_t* next_channel_id;
    ReshardCache* reshard_cache;
    SpmdPartitioner* partitioner;
  };
  PartitionedHlo(HloInstruction* hlo, Shape base_shape, PartitioningState state)
      : hlo_(hlo), base_shape_(base_shape), state_(std::move(state)) {
    CHECK(hlo->has_sharding())
        << "PartitionedHlo is missing sharding:" << hlo->ToString();
  }

  PartitionedHlo CloneWithNewHlo(HloInstruction* hlo) const {
    PartitionedHlo new_phlo = *this;
    new_phlo.hlo_ = hlo;
    if (!hlo->has_sharding() && hlo_->has_sharding()) {
      hlo->copy_sharding(hlo_);
    }
    return new_phlo;
  }

  // Reshards the current SPMD instruction to a new sharding with optional
  // specified pad value used during resharding. Could only modify the reshard
  // cache.
  PartitionedHlo Reshard(const HloSharding& target,
                         std::optional<Literal> pad_value = std::nullopt) const;

  // Pads the garbage area of the output with the provided value. Normally,
  // unevenly partitioned dimensions are padded on the right, but this function
  // allows specifying left-padded dimensions, which can be used during the
  // handling of kReverse, etc.
  PartitionedHlo PadWithValue(
      HloInstruction* pad_value,
      absl::Span<const int64_t> left_padded_dims = {},
      absl::Span<const int64_t> skipped_dims = {}) const;

  // Same as PadWithValue but does not create a new PartitionedHlo.
  HloInstruction* PadWithValueHlo(
      HloInstruction* pad_value,
      absl::Span<const int64_t> left_padded_dims = {},
      absl::Span<const int64_t> skipped_dims = {}) const;

  // Same as PadWithValue with zero as the pad value.
  PartitionedHlo PadWithZero(absl::Span<const int64_t> left_padded_dims = {},
                             absl::Span<const int64_t> skipped_dims = {}) const;

  // PadWithZero consider all dimensions except the skipped dimensions.
  // PadWithZeroOnSpecifiedDims considers only the specified dimensions.
  PartitionedHlo PadWithZeroOnSpecifiedDims(
      absl::Span<const int64_t> dims,
      absl::Span<const int64_t> left_padded_dims = {}) const;

  // Returns the SPMD instruction.
  HloInstruction* hlo() const { return hlo_; }

  // Returns the sharding of the SPMD instruction.
  const HloSharding& sharding() const { return hlo_->sharding(); }

  // Returns the SPMD instruction's number of dimensions.
  int64_t num_dimensions() const { return base_shape_.dimensions().size(); }

  // Original full shape of the data.
  const Shape& base_shape() const { return base_shape_; }

  int64_t NewChannel() const { return (*state_.next_channel_id)++; }

  // Reshards the HLO to a usable partitioned input for a windowed user. Could
  // only modify the reshard cache.
  std::optional<WindowedInputShardReturnValue> ReshardAsWindowedInput(
      const Window& window, const HloSharding& target,
      HloInstruction* pad_value, bool mask_invalid_region = true,
      bool force_mask_in_compact = false);

  const PartitioningState& state() const { return state_; }

  void AddReshardCache(const HloSharding& sharding, const PartitionedHlo& phlo);

  // Helper function to replicate the data on all devices. Could only modify
  // the reshard cache.
  PartitionedHlo Replicate() const;

  // Helper function to replicate the data for partitions along the given dims.
  HloInstruction* ReplicatePartial(absl::Span<const int64_t> dims) const;

  // Set state of the partitoned HLO.
  void set_state(PartitioningState state) { state_ = std::move(state); }

 private:
  // Same as Reshard except that it does not explicitly modify the reshard
  // cache, although it would indirectly modify by calling Replicate().
  PartitionedHlo ReshardNoCache(const HloSharding& target,
                                std::optional<Literal> pad_value = std::nullopt,
                                bool allow_full_replication = true) const;

  // Helper function to broadcast data from a single device to all devices.
  PartitionedHlo Broadcast() const;

  // Try to perform complicated reshard handling by splitting a big reshard into
  // multiple reshards using that can be handled directly.
  std::optional<PartitionedHlo> TryComplexReshardHandling(
      const HloSharding& target) const;

  // Helper function to reshard the tensor using AllToAll (instead of the
  // default of Replicate followed by Slice).
  PartitionedHlo ReshardWithAllToAll(
      const HloSharding& target,
      absl::Span<const std::pair<int64_t, int64_t>> source_target_dims,
      bool try_multiple_source_target_dims = true) const;

  // Called by ReshardWithAllToAll if try_multiple_source_target_dims is true.
  // Try to handle multiple source and target dims in a single AllToAll.
  PartitionedHlo TryMultipleSourceTargetDims(
      const HloSharding& target,
      absl::Span<const std::pair<int64_t, int64_t>> source_target_dims) const;

  // Helper function to reshard the tensor using CollectivePermute.
  PartitionedHlo ReshardWithCollectivePermute(const HloSharding& target) const;

  // Helper function to reshard to partial replicate using AllGather.
  std::optional<PartitionedHlo> ReshardToPartialReplicateWithAllGather(
      const HloSharding& target) const;

  // Helper function to reshard from partial replicate using DynamicSlice.
  std::optional<PartitionedHlo> ReshardFromPartialReplicateWithDynamicSlice(
      const HloSharding& target) const;

  // Helper function to reshard from partial replicate using AllToAll.
  std::optional<PartitionedHlo> ReshardPartialReplicateWithAllToAll(
      const HloSharding& target) const;

  // SPMD instruction.
  HloInstruction* hlo_;

  // The original shape of the data before SPMD transformation is applied.
  Shape base_shape_;

  PartitioningState state_;
};

class SpmdPartitioningVisitor : public DfsHloVisitorWithDefault {
 public:
  SpmdPartitioningVisitor(
      HloComputation* computation, int64_t num_partitions, int64_t num_replicas,
      const SPMDCollectiveOpsCreator& collective_ops_creator,
      int64_t* next_channel_id, SpmdLogger* logger,
      SpmdPartitionerOptions options, SpmdPartitioner* partitioner,
      const CallGraph& call_graph);

  SpmdPartitioningVisitor(const SpmdPartitioningVisitor& src);

  absl::Status DefaultAction(HloInstruction* hlo) override;

  absl::Status HandleAllReduce(HloInstruction* hlo) override;
  absl::Status HandleBitcastConvert(HloInstruction* hlo) override;
  absl::Status HandleBroadcast(HloInstruction* hlo) override;
  absl::Status HandleCall(HloInstruction* hlo) override;
  absl::Status HandleCholesky(HloInstruction* hlo) override;
  absl::Status HandleCollectivePermute(HloInstruction* hlo) override;
  absl::Status HandleConcatenate(HloInstruction* hlo) override;
  absl::Status HandleConditional(HloInstruction* hlo) override;
  absl::Status HandleConstant(HloInstruction* hlo) override;
  absl::Status HandleConvolution(HloInstruction* hlo) override;
  absl::Status HandleCustomCall(HloInstruction* hlo) override;
  absl::Status HandleDot(HloInstruction* hlo) override;
  absl::Status HandleDynamicSlice(HloInstruction* hlo) override;
  absl::Status HandleDynamicUpdateSlice(HloInstruction* hlo) override;
  absl::Status HandleFft(HloInstruction* hlo) override;
  absl::Status HandleGather(HloInstruction* hlo) override;
  absl::Status HandleGetTupleElement(HloInstruction* hlo) override;
  absl::Status HandleInfeed(HloInstruction* hlo) override;
  absl::Status HandleIota(HloInstruction* hlo) override;
  absl::Status HandleOptimizationBarrier(HloInstruction* hlo) override;
  absl::Status HandleOutfeed(HloInstruction* hlo) override;
  absl::Status HandlePad(HloInstruction* hlo) override;
  absl::Status HandleParameter(HloInstruction* hlo) override;
  absl::Status HandlePartitionId(HloInstruction* hlo) override;
  absl::Status HandleRaggedDot(HloInstruction* hlo) override;
  absl::Status HandleReduce(HloInstruction* hlo) override;
  absl::Status HandleReduceWindow(HloInstruction* hlo) override;
  absl::Status HandleReshape(HloInstruction* hlo) override;
  absl::Status HandleReverse(HloInstruction* hlo) override;
  absl::Status HandleRng(HloInstruction* hlo) override;
  absl::Status HandleScatter(HloInstruction* hlo) override;
  absl::Status HandleSelectAndScatter(HloInstruction* hlo) override;
  absl::Status HandleSlice(HloInstruction* hlo) override;
  absl::Status HandleSort(HloInstruction* hlo) override;
  absl::Status HandleTranspose(HloInstruction* hlo) override;
  absl::Status HandleTriangularSolve(HloInstruction* hlo) override;
  absl::Status HandleTuple(HloInstruction* hlo) override;
  absl::Status HandleWhile(HloInstruction* hlo) override;

  // Implementation of dot partitioning given DotGeneralDimsMapping.
  absl::Status HandleDotHelper(
      HloInstruction* hlo,
      const dot_as_convolution_util::DotConvolutionDimsInfo& dims_mapping,
      absl::FunctionRef<absl::StatusOr<HloInstruction*>(
          HloInstruction*, HloInstruction*, SpmdBuilder*,
          const Window& conv_window)>
          create_sharded_dot);

  // Common handle for elementwise HLOs.
  absl::Status HandleElementwise(HloInstruction* hlo);

  // All dimensions in the hlo are element-wise except that we replicate
  // `dims_to_replicate`.
  absl::Status HandleElementwiseWithDimsToReplicate(
      HloInstruction* hlo, absl::Span<const int64_t> dims_to_replicate);

  // Common handle for HLOs that runs on a single device.
  absl::Status HandleSingleDevice(const HloInstruction* hlo);

  // CustomCall handlers per call target.
  absl::Status HandleCustomCallTopK(HloInstruction* hlo);
  // Convenient custom ops defined by the partitioner itself.
  absl::Status HandleCustomCallSPMDInternal_RotateRight(HloInstruction* hlo);

  virtual std::unique_ptr<SpmdPartitioningVisitor> Clone() const;

  // Returns the PartitionedHlo that corresponds to the original hlo.
  PartitionedHlo& GetPartitionedHlo(const HloInstruction* hlo) {
    CHECK_EQ(partitioned_instructions_.count(hlo), 1);
    return partitioned_instructions_.find(hlo)->second;
  }

  // Sets the PartitionedHlo for the original hlo.
  void SetPartitionedHlo(const HloInstruction* hlo,
                         const PartitionedHlo& partitioned_hlo) {
    CHECK_EQ(partitioned_instructions_.count(hlo), 0);
    partitioned_instructions_.emplace(hlo, partitioned_hlo);
    changed_ = true;
  }

  // Convenient wrapper that creates PartitionedHlo from the result of the func
  // and maps it to the given original hlo.
  void SetPartitionedHlo(const HloInstruction* hlo,
                         absl::FunctionRef<HloInstruction*()> func) {
    HloInstruction* new_hlo = func();
    new_hlo->set_sharding(hlo->sharding());
    SetPartitionedHlo(
        hlo, PartitionedHlo(new_hlo, hlo->shape(), MakePartitioningState()));
    changed_ = true;
  }

  int64_t NewChannel() { return (*next_channel_id_)++; }

  PartitionedHlo::PartitioningState MakePartitioningState();

  SpmdBuilder* builder() { return &b_; }

  virtual absl::StatusOr<bool> DoPartition(
      HloComputation* computation, const HloSharding& root_sharding,
      const SpmdPartitionerOptions& options);

  virtual double GetComputationTimeInMilliSec(HloInstruction* hlo) {
    return 0.0;
  }

  virtual double GetCommunicationTimeInMilliSec(
      int64_t bytes, absl::Span<const ReplicaGroup> device_groups) {
    return 0.0;
  }

  virtual int GetCommunicationMultiplier(
      absl::Span<const ReplicaGroup> device_groups) {
    return 1;
  }

  std::vector<ReplicaGroup> CreateReplicaGroups(
      std::vector<std::vector<int64_t>>& groups);

  const CallGraph& call_graph() { return call_graph_; }
  int64_t num_partitions() const { return num_partitions_; }
  int64_t num_replicas() const { return num_replicas_; }
  SpmdLogger* logger() { return logger_; }
  const SpmdLogger* logger() const { return logger_; }
  const SpmdPartitionerOptions& options() const { return options_; }
  SpmdPartitioner* partitioner() { return partitioner_; }
  const SpmdPartitioner* partitioner() const { return partitioner_; }
  SPMDCollectiveOpsCreator& collective_ops_creator() {
    return collective_ops_creator_;
  }
  const SPMDCollectiveOpsCreator& collective_ops_creator() const {
    return collective_ops_creator_;
  }
  HloModule* module() { return module_; }
  const HloModule* module() const { return module_; }
  void set_module(HloModule* module) { module_ = module; }

  // Information about a loop created for windowed dot-general. Used when
  // DoCodeMotionForWindowedDotGeneralLoops() executes after the visitor
  // finishes traversing the graph.
  struct WindowedDotGeneralLoop {
    HloInstruction* while_loop;
    int64_t windowed_operand;
    bool windowed_in_contracting_dims;
    bool windowed_in_batch_dims;
    bool operands_sharded_at_contracting_dims;
    int64_t num_partitions;
    std::vector<ReplicaGroup> loop_replica_groups;
  };

 protected:
  absl::Status Preprocess(HloInstruction* hlo) override;
  absl::Status Postprocess(HloInstruction* hlo) override;

  // Performs code motion for windowed dot-general loops in
  // windowed_dot_general_loops_. Invoked after the visitor finishes traversing
  // the graph.
  absl::Status DoCodeMotionForWindowedDotGeneralLoops(
      HloComputation* computation, const SpmdPartitionerOptions& options);

  bool changed_;
  HloModule* module_;
  int64_t num_partitions_;
  int64_t num_replicas_;

  SPMDCollectiveOpsCreator collective_ops_creator_;

  // Tracks the next channel id to use for cross-partition all-reduce.
  int64_t* next_channel_id_;
  SpmdBuilder b_;

  std::vector<WindowedDotGeneralLoop> windowed_dot_general_loops_;

  HloInstruction* partition_id_;

 private:
  PartitionedHlo::ReshardCache reshard_cache_;

  // Mapping from the instruction in the original computation to the new SPMD
  // partitioned instruction.
  ConstHloInstructionMap<PartitionedHlo> partitioned_instructions_;

  HloInstruction* visiting_hlo_;
  SpmdLogger* logger_;
  const SpmdPartitionerOptions options_;
  SpmdPartitioner* partitioner_;
  std::vector<HloSharding> visiting_hlo_operand_shardings_;
  std::optional<HloSharding> visiting_hlo_sharding_;
  std::optional<int64_t> visiting_num_partitions_;
  std::optional<SPMDCollectiveOpsCreator> visiting_collective_ops_creator_;
  std::optional<HloInstruction*> visiting_partition_id_;
  std::vector<PartitionedHlo::PartitioningState> visiting_state_;
  std::vector<std::vector<int64_t>> device_groups_;
  const CallGraph& call_graph_;
};

}  // namespace spmd
}  // namespace xla
#endif  // XLA_SERVICE_SPMD_SPMD_PARTITIONER_H_
