/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_SPMD_PARTITIONER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_SPMD_PARTITIONER_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"

namespace xla {
namespace spmd {

struct SpmdPartitionerOptions {
  // Always exchange halo on LHS for all convolutions. If false, backprop filter
  // convolution exchanges halo on RHS.
  bool conv_halo_exchange_always_on_lhs = true;

  // The number of instructions to be reported for the highest memory profile
  // instructions.
  int64 report_instruction_count = 5;

  // The minimum size in MiB of an einsum operand to be considered using
  // windowed implementation in an HLO loop.
  int64 threshold_for_windowed_einsum_mib = 256;

  // Whether the entry computations' signature could change after partitioning.
  bool allow_module_signature_change = false;
};

// Class to wrap the computation builder to capture information during SPMD
// transformation.
class SpmdBuilder : public HloComputation::Builder {
 public:
  SpmdBuilder(const std::string& name, HloInstruction* hlo)
      : HloComputation::Builder(name) {
    visiting_hlo_ = hlo;
  }
  HloInstruction* AddInstruction(std::unique_ptr<HloInstruction> instruction);

  const std::vector<HloInstruction*>& derived_instructions(
      HloInstruction* hlo) {
    return instructions_.at(hlo);
  }

  void set_visiting_hlo(HloInstruction* hlo) { visiting_hlo_ = hlo; }

  HloInstruction* visiting_hlo() const { return visiting_hlo_; }

 private:
  // Currently visiting instruction.
  HloInstruction* visiting_hlo_;

  // Map from the currently visiting (old) instruction to new instructions
  // created during SPMD partitioning.
  HloInstructionMap<std::vector<HloInstruction*>> instructions_;
};

// A set of functions that create the cross-partition collective ops.
struct SPMDCollectiveOpsCreator {
  // Function used to create a partition ID HLO.
  std::function<HloInstruction*(SpmdBuilder*)> create_partition_id;

  // Function used to create a cross-partition all-reduce HLO.
  std::function<HloInstruction*(SpmdBuilder*, HloInstruction* operand,
                                HloComputation* reduction, int64 channel_id)>
      create_cross_partition_all_reduce;

  // Function used to create a cross-partition collective-permute HLO.
  std::function<HloInstruction*(
      SpmdBuilder*, HloInstruction* operand,
      std::vector<std::pair<int64, int64>>& src_dst_pairs,
      int64 next_channel_id)>
      create_cross_partition_collective_permute;

  // Function used to create a cross-partition all-to-all HLO.
  std::function<HloInstruction*(
      SpmdBuilder*, absl::Span<HloInstruction* const> operands,
      const std::vector<ReplicaGroup>& replica_groups, int64 channel_id,
      absl::optional<int64> split_dimension)>
      create_cross_partition_all_to_all;

  // Function used to create a cross-partition all-gather HLO. This is optional:
  // if it is nullptr, the partitioner will use all-reduce instead.
  std::function<HloInstruction*(
      SpmdBuilder*, HloInstruction* operand, const Shape& ag_shape,
      const std::vector<std::vector<int64>>& partition_subgroups,
      int64 channel_id, int64 all_gather_dimension)>
      create_cross_partition_all_gather;
};

// Create a default SPMDCollectiveOpsCreator.
SPMDCollectiveOpsCreator GetDefaultCollectiveOpsCreator(int64 num_partitions,
                                                        int64 num_replicas);

// Logger to report memory usage during SPMD partitioning.
class SpmdLogger {
 public:
  explicit SpmdLogger(int64 report_instruction_count)
      : report_instruction_count_(report_instruction_count) {}
  static std::string ReportBeforePartition(const HloModule& module,
                                           int64 report_instruction_count);
  static std::string ReportAfterPartition(const HloModule& module,
                                          int64 report_instruction_count);

  // Registers the logging for the groups of instructions created to transform
  // the given hlo.
  void RegisterLogEntry(HloInstruction* hlo,
                        const std::vector<HloInstruction*>& group);

  std::string MakeReport();

 private:
  template <typename F>
  static std::string ReportMemoryUsage(const HloModule& module, const F& filter,
                                       int64 report_instruction_count);

  // A vector of logging messages (one for each original HLO instruction), where
  // the first integer of the pair represents the size of the HBM used.
  std::vector<std::pair<int64, std::string>> entries_;

  int64 report_instruction_count_;
};

class SpmdPartitioningVisitor;

class SpmdPartitioner : public HloModulePass {
 public:
  SpmdPartitioner(int64 num_partitions, int64 num_replicas,
                  SpmdPartitionerOptions options);
  SpmdPartitioner(int64 num_partitions, int64 num_replicas,
                  SpmdPartitionerOptions options,
                  SPMDCollectiveOpsCreator collective_ops_creator)
      : num_partitions_(num_partitions),
        num_replicas_(num_replicas),
        options_(std::move(options)),
        collective_ops_creator_(std::move(collective_ops_creator)) {}
  absl::string_view name() const override { return "spmd-partitioning"; }
  StatusOr<bool> Run(HloModule* module) override;

  // Transforms the given computation with SPMD instructions, replacing it with
  // a new computation.
  StatusOr<bool> PartitionComputation(HloComputation* computation,
                                      const HloSharding& root_sharding,
                                      int64* next_channel_id,
                                      SpmdLogger* logger);

  // Creates all-gather based on HloSharding. Can be overridden to customize.
  // The default uses a single all-gather even if there are multiple sharded
  // dimensions, and adds potential reshapes and transposes to achieve that.
  // If it returns false, the partitioner will fall back to all-reduce.
  virtual HloInstruction* AllGatherShards(SpmdBuilder* b,
                                          HloInstruction* operand,
                                          const HloSharding& sharding,
                                          int64 channel_id);

 protected:
  virtual std::unique_ptr<SpmdPartitioningVisitor> CreateVisitor(
      HloComputation* computation, int64 num_partitions, int64 num_replicas,
      const SPMDCollectiveOpsCreator& collective_ops_creator,
      int64* next_channel_id, SpmdLogger* logger,
      SpmdPartitionerOptions options);

  // Verify that the sharding of instructions in the module are valid, and also
  // fill in missing sharding information.
  Status PreprocessSharding(HloModule* module);

  const int64 num_partitions_;
  const int64 num_replicas_;

  SpmdPartitionerOptions options_;
  SPMDCollectiveOpsCreator collective_ops_creator_;
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
    absl::optional<std::vector<HloInstruction*>> dynamic_slice_index_on_output;
  };
  // A cache for resharding each partitioned HLO.
  struct ReshardCache {
    struct PerHloCache {
      std::vector<std::pair<HloSharding, PartitionedHlo>> reshard_cache;
      std::vector<
          std::tuple<HloSharding, Window, WindowedInputShardReturnValue>>
          window_reshard_cache;
    };
    std::unordered_map<HloInstruction*, PerHloCache> per_hlo_cache;
  };
  struct PartitioningState {
    SpmdBuilder* b;
    HloModule* module;
    int64 num_replicas;
    HloInstruction* partition_id;
    SPMDCollectiveOpsCreator collective_ops_creator;
    int64* next_channel_id;
    ReshardCache* reshard_cache;
    SpmdPartitioner* partitioner;
  };
  PartitionedHlo(HloInstruction* hlo, Shape base_shape, PartitioningState state)
      : hlo_(hlo), base_shape_(base_shape), state_(std::move(state)) {
    CHECK(hlo->has_sharding())
        << "PartitionedHlo is missing sharding:" << hlo->ToString();
    // If the tuple shape instruction does not have a tuple sharding, reassign
    // to use the tuple sharding. Reshard() implementation assumes this.
    if (hlo_->shape().IsTuple() && !hlo_->sharding().IsTuple()) {
      hlo_->set_sharding(
          hlo_->sharding().GetTupleSharding(hlo_->shape()).ValueOrDie());
    }
  }

  // Reshards the current SPMD instruction to a new sharding. Could only modify
  // the reshard cache.
  PartitionedHlo Reshard(const HloSharding& target);

  // Pads the garbage area of the output with the provided value.
  PartitionedHlo PadWithValue(HloInstruction* pad_value) const;

  // Returns the SPMD instruction.
  HloInstruction* hlo() const { return hlo_; }

  // Returns the sharding of the SPMD instruction.
  const HloSharding& sharding() const { return hlo_->sharding(); }

  // Original full shape of the data.
  const Shape& base_shape() const { return base_shape_; }

  int64 NewChannel() const { return (*state_.next_channel_id)++; }

  // Reshards the HLO to a usable partitioned input for a windowed user. Could
  // only modify the reshard cache.
  absl::optional<WindowedInputShardReturnValue> ReshardAsWindowedInput(
      const Window& window, const HloSharding& target,
      HloInstruction* pad_value, bool mask_invalid_region = true);

 private:
  // Same as Reshard except that it does not explicitly modify the reshard
  // cache, although it would indirectly modify by calling Replicate().
  PartitionedHlo ReshardNoCache(const HloSharding& target);

  // Helper function to replicate the data on all devices. Could only modify
  // the reshard cache.
  PartitionedHlo Replicate();

  // Helper function to broadcast data from a single device to all devices.
  PartitionedHlo Broadcast() const;

  // Helper function to reshard the tensor using AllToAll (instead of the
  // default of Replicate followed by Slice).
  PartitionedHlo ReshardWithAllToAll(const HloSharding& target) const;

  // Helper function to reshard the tensor using CollectivePermute.
  PartitionedHlo ReshardWithCollectivePermute(const HloSharding& target) const;

  // SPMD instruction.
  HloInstruction* hlo_;

  // The original shape of the data before SPMD transformation is applied.
  Shape base_shape_;

  PartitioningState state_;
};

struct DotGeneralDimsMapping {
  // The dimension numbers for the operands and output corresponding to a
  // logical dimension (e.g., batch, contracting, non-contracting). If an
  // operand or the output doesn't have the logical dimension, it is set to
  // -1.
  struct DimsMapping {
    int64 lhs;
    int64 rhs;
    int64 output;
  };
  std::vector<DimsMapping> batch_dims;
  std::vector<DimsMapping> contracting_dims;
  std::vector<DimsMapping> lhs_non_contracting_dims;
  std::vector<DimsMapping> rhs_non_contracting_dims;
};

class SpmdPartitioningVisitor : public DfsHloVisitorWithDefault {
 public:
  SpmdPartitioningVisitor(
      HloComputation* computation, int64 num_partitions, int64 num_replicas,
      const SPMDCollectiveOpsCreator& collective_ops_creator,
      int64* next_channel_id, SpmdLogger* logger,
      SpmdPartitionerOptions options, SpmdPartitioner* partitioner);

  Status DefaultAction(HloInstruction* hlo) override;
  Status HandleAllReduce(HloInstruction* hlo) override;
  Status HandleBroadcast(HloInstruction* hlo) override;
  Status HandleConstant(HloInstruction* hlo) override;
  Status HandleCustomCall(HloInstruction* hlo) override;
  Status HandleDot(HloInstruction* hlo) override;
  Status HandleDynamicSlice(HloInstruction* hlo) override;
  Status HandleDynamicUpdateSlice(HloInstruction* hlo) override;
  Status HandleGather(HloInstruction* hlo) override;
  Status HandleGetTupleElement(HloInstruction* hlo) override;
  Status HandleInfeed(HloInstruction* hlo) override;
  Status HandleOutfeed(HloInstruction* hlo) override;
  Status HandlePad(HloInstruction* hlo) override;
  Status HandleParameter(HloInstruction* hlo) override;
  Status HandleReduce(HloInstruction* hlo) override;
  Status HandleReverse(HloInstruction* hlo) override;
  Status HandleWhile(HloInstruction* hlo) override;
  Status HandleConditional(HloInstruction* hlo) override;
  Status HandleReduceWindow(HloInstruction* hlo) override;
  Status HandleSelectAndScatter(HloInstruction* hlo) override;
  Status HandleTuple(HloInstruction* hlo) override;
  Status HandleRng(HloInstruction* hlo) override;
  Status HandleConvolution(HloInstruction* hlo) override;
  Status HandleConcatenate(HloInstruction* hlo) override;
  Status HandleScatter(HloInstruction* hlo) override;
  Status HandleSlice(HloInstruction* hlo) override;
  Status HandleSort(HloInstruction* hlo) override;
  Status HandleTranspose(HloInstruction* hlo) override;
  Status HandleReshape(HloInstruction* hlo) override;
  Status HandleIota(HloInstruction* hlo) override;
  Status HandlePartitionId(HloInstruction* hlo) override;

  // Handles convolution where both LHS and RHS operands are tiled.
  Status HandleConvolutionTiledLhsAndRhs(HloInstruction* hlo);

  // Implementation of dot partitioning given DotGeneralDimsMapping.
  Status HandleDotHelper(
      HloInstruction* hlo, const DotGeneralDimsMapping& dims_mapping,
      const std::function<StatusOr<HloInstruction*>(
          HloInstruction*, HloInstruction*, SpmdBuilder*)>& create_sharded_dot);

  // Common handle for elementwise HLOs.
  Status HandleElementwise(HloInstruction* hlo);

  // Common handle for HLOs that runs on a single device.
  Status HandleSingleDevice(const HloInstruction* hlo);

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
                         const std::function<HloInstruction*()>& func) {
    HloInstruction* new_hlo = func();
    new_hlo->set_sharding(hlo->sharding());
    new_hlo->set_metadata(hlo->metadata());
    SetPartitionedHlo(
        hlo, PartitionedHlo(new_hlo, hlo->shape(), MakePartitioningState()));
    changed_ = true;
  }

  int64 NewChannel() { return (*next_channel_id_)++; }

  PartitionedHlo::PartitioningState MakePartitioningState() {
    PartitionedHlo::PartitioningState state;
    state.b = &b_;
    state.module = module_;
    state.num_replicas = num_replicas_;
    state.partition_id = partition_id_;
    state.collective_ops_creator = collective_ops_creator_;
    state.next_channel_id = next_channel_id_;
    state.reshard_cache = &reshard_cache_;
    state.partitioner = partitioner_;
    return state;
  }

  SpmdBuilder* builder() { return &b_; }

  StatusOr<bool> DoPartition(HloComputation* computation,
                             const HloSharding& root_sharding);

 private:
  Status Preprocess(HloInstruction* hlo) override;
  Status Postprocess(HloInstruction* hlo) override;

  // Performs code motion for windowed dot-general loops in
  // windowed_dot_general_loops_. Invoked after the visitor finishes traversing
  // the graph.
  Status DoCodeMotionForWindowedDotGeneralLoops(HloComputation* computation);

  bool changed_;
  HloModule* module_;
  int64 num_partitions_;
  int64 num_replicas_;

  SPMDCollectiveOpsCreator collective_ops_creator_;

  // Tracks the next channel id to use for cross-partition all-reduce.
  int64* next_channel_id_;
  SpmdBuilder b_;

  HloInstruction* partition_id_;

  PartitionedHlo::ReshardCache reshard_cache_;

  // Mapping from the instruction in the original computation to the new SPMD
  // partitioned instruction.
  ConstHloInstructionMap<PartitionedHlo> partitioned_instructions_;

  // Information about a loop created for windowed dot-general. Used when
  // DoCodeMotionForWindowedDotGeneralLoops() executes after the visitor
  // finishes traversing the graph.
  struct WindowedDotGeneralLoop {
    HloInstruction* while_loop;
    int64 windowed_operand;
    bool windowed_in_contracting_dims;
    bool windowed_in_batch_dims;
  };
  std::vector<WindowedDotGeneralLoop> windowed_dot_general_loops_;

  HloInstruction* visiting_hlo_;
  SpmdLogger* logger_;
  const SpmdPartitionerOptions options_;
  SpmdPartitioner* partitioner_;
};

}  // namespace spmd
}  // namespace xla
#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SPMD_SPMD_PARTITIONER_H_
