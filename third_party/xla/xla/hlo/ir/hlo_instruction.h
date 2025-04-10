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

// HLO instructions are in DAG form and represent the computations that the user
// has built up via the XLA service interface. They are ultimately lowered
// in a platform-aware way by traversing the HLO DAG and emitting a lowered
// form; e.g. see DfsHloVisitor.

#ifndef XLA_HLO_IR_HLO_INSTRUCTION_H_
#define XLA_HLO_IR_HLO_INSTRUCTION_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/backend_config.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_domain_metadata.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/ptrvec.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/printer.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/mapped_ptr_container_sorter.h"
#include "xla/service/name_uniquer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/gtl/iterator_range.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {

class HloComputation;
class HloModule;
class HloInstruction;

// A small holder that is used to keep some immutable info alongside an
// instruction pointer in an HloComputation's list of instructions
class HloInstructionInfo {
 public:
  HloInstruction* get() const { return inst_; }
  HloInstruction& operator*() { return *inst_; }
  HloInstruction* operator->() { return inst_; }
  const HloInstruction& operator*() const { return *inst_; }
  const HloInstruction* operator->() const { return inst_; }

  HloOpcode opcode() const { return opcode_; }
  HloInstruction* inst() const { return inst_; }

 private:
  friend class HloComputation;
  HloOpcode opcode_;
  HloInstruction* inst_;
};

namespace mapped_ptr_container_sorter_internal {

template <typename T>
struct PtrGetter<const HloInstructionInfo&, const T*> {
  static const T* Get(const HloInstructionInfo& p) { return p.get(); }
};

}  // namespace mapped_ptr_container_sorter_internal

using HloInstructionList = std::vector<HloInstructionInfo>;

template <typename UnderlyingList>
class HloInstructionIteratorBase {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = HloInstructionInfo;
  using difference_type = ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;

  HloInstructionIteratorBase(UnderlyingList* list, int begin_index,
                             int end_index)
      : list_(list), current_(begin_index), end_index_(end_index) {
    if (current_ < end_index_ && (*list_)[current_].inst() == nullptr) {
      ++*this;
    }
  }

  HloInstruction* get() const { return (*list_)[current_].inst(); }

  auto operator*() -> HloInstructionInfo { return (*list_)[current_]; }
  HloInstructionIteratorBase& operator++() {
    int next = current_;
    do {
      ++next;
    } while (next < end_index_ && (*list_)[next].inst() == nullptr);
    current_ = next;
    return *this;
  }
  HloInstructionIteratorBase operator++(int) {
    HloInstructionIteratorBase temp(list_, current_, end_index_);
    operator++();
    return temp;
  }

  friend bool operator==(const HloInstructionIteratorBase& a,
                         const HloInstructionIteratorBase& b) {
    return a.current_ == b.current_;
  }

  friend bool operator!=(const HloInstructionIteratorBase& a,
                         const HloInstructionIteratorBase& b) {
    return !(a == b);
  }

 private:
  UnderlyingList* list_;
  int current_;
  int end_index_;
};
using HloInstructionIterator = HloInstructionIteratorBase<HloInstructionList>;
using HloInstructionConstIterator =
    HloInstructionIteratorBase<const HloInstructionList>;

template <typename WrappedIter>
class HloInstructionUnwrappingIteratorBase {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = HloInstruction*;
  using difference_type = ptrdiff_t;
  using pointer = value_type*;
  using reference = value_type&;

  explicit HloInstructionUnwrappingIteratorBase(WrappedIter iter)
      : iter_(std::move(iter)) {}

  auto operator*() -> value_type { return iter_.get(); }
  HloInstructionUnwrappingIteratorBase& operator++() {
    ++iter_;
    return *this;
  }
  HloInstructionUnwrappingIteratorBase operator++(int) {
    HloInstructionUnwrappingIteratorBase temp(iter_);
    operator++();
    return temp;
  }

  friend bool operator==(const HloInstructionUnwrappingIteratorBase& a,
                         const HloInstructionUnwrappingIteratorBase& b) {
    return a.iter_ == b.iter_;
  }

  friend bool operator!=(const HloInstructionUnwrappingIteratorBase& a,
                         const HloInstructionUnwrappingIteratorBase& b) {
    return !(a == b);
  }

 private:
  WrappedIter iter_;
};
using HloInstructionUnwrappingIterator =
    HloInstructionUnwrappingIteratorBase<HloInstructionIterator>;
using HloInstructionUnwrappingConstIterator =
    HloInstructionUnwrappingIteratorBase<HloInstructionConstIterator>;

// HLO instructions are the atomic unit of the high-level compiler's IR.
//
// HloInstructions live inside of an HloComputation, which is analogous to a
// function in other programming languages.  Nodes have no total order within
// their computation.  Instead, they have a partial ordering determined by their
// data and control dependencies.
//
// HLO does not have basic blocks or explicit "branch" instructions.  Instead,
// certain HloInstructions -- namely, kWhile, kConditional, and kCall -- encode
// control flow.  For example, the kConditional HLO executes one of two possible
// computations, depending on the runtime value of a predicate.
//
// HLO is pure (mostly).  It has no concept of mutable state.  Instead, data
// values are produced by one HLO and flow into consumers across dependency
// edges.
class HloInstruction {
 public:
  // A fusion node computes the same value a call to its fusion computation
  // would compute.  However, the choice of fusion kind dictates codegen
  // strategy for the backend.
  //
  // To generate code for a kFusion HloInstruction, most backends do something
  // like the following:
  //
  // 1) Identify the "primary" HloInstruction of the fused computation.
  // 2) Emit code that does the work of the primary node, creating its inputs
  //    and transforming its outputs as specified by the fused computation.
  //
  // In step (2), the code emitted is usually similar to the code that would be
  // emitted for an *unfused* version of the primary node, except that
  //
  //  - when the primary node reads an element of one of its operands, instead
  //    of loading the value from memory, it *computes* the value based on the
  //    contents of the fused computation.
  //  - when the primary node outputs a value, instead of storing it to memory,
  //    it forwards the value to its users, which then perform additional
  //    computations before the value is finally stored to memory at the root of
  //    the fusion node.
  //
  // An HloInstruction's FusionKind helps us find the kFusion instruction's
  // primary node, and can also affect how we generate code in step (2).
  //
  //  - kInput: The primary node is the root of the fused instruction.
  //
  //  - kOutput: The primary node is not the root of the fused instruction.
  //    This fusion kind requires that one operand buffer of the fusion
  //    instruction be able to alias the output buffer.  This constraint is
  //    usually enough to let backends find the primary node unambiguously.
  //
  //  - kLoop: The primary node is the root of the fused computation, but,
  //    unlike in input fusion, we prescribe a specific implementation for
  //    codegen.  Rather than generating code that looks like the code we'd emit
  //    for an unfused version of the primary/root node, we emit code that
  //    generates one element of the root at a time.
  //
  //  - kCustom: Custom category for backend-specific fusions that don't fit
  //    into the above patterns.
  //
  // Not all backends support all fusion kinds, and given a particular fused
  // computation, it's not in general safe to change its fusion kind.  Creation
  // of fusion nodes is always backend-specific.
  //
  // For elementwise ops (e.g. kAdd), most backends would emit a
  // one-element-at-a-time implementation for the unfused version, so loop
  // fusion and input fusion are probably equivalent if the root node is
  // elementwise.  They're not necessarily equivalent e.g. for kReduce, where an
  // implementation might emit something more sophisticated for an unfused or
  // input-fusion reduce, but will emit the naive code that reduces one element
  // at a time for loop fusion with a reduce as the root.
  //
  // Another way to think of loop fusion is that it's equivalent to input
  // fusion, but where the root node is an implicit identity node, whose
  // unfused implementation is "read one element, write one element".
  //
  // TODO(b/79869434): This categorization scheme is not great.  For one thing,
  // input and loop fusion are basically the same thing: There is no reason for
  // the HLO to encode backend-specific decisions about how e.g. a reduce that's
  // the root of a fusion should be lowered.  In addition, this scheme as
  // written doesn't work for multi-output fusion, where the primary node is
  // never actually the root (which is a kTuple instruction that gathers the
  // multiple outputs of the fusion).
  enum class FusionKind {
    kLoop,
    kInput,
    kOutput,
    kCustom,
  };

  inline static constexpr char kMainExecutionThread[] = "main";
  inline static constexpr char kHostThread[] = "host";

  virtual ~HloInstruction() { DetachFromOperandsAndUsers(); }

  // Detaches an instruction from its operands and users. That is, remove the
  // instruction from each operand's user set and user's operand set.
  void DetachFromOperandsAndUsers();

  // Adds a derived instruction to the parent computation of this instruction.
  // Updates setup the new instruction as a derived instruction, and sets the
  // name of the new instruction (if `new_name` is not empty).
  HloInstruction* AddInstruction(
      std::unique_ptr<HloInstruction> derived_instruction,
      absl::string_view new_name = "");

  // Creates an instruction from the given proto. Arguments:
  //
  //   proto: the proto to convert from.
  //   instruction_map: a map from instruction id to HloInstruction*. This map
  //     must contain all operands of the newly constructed instruction.
  //   computation_map: a map from computation id to HloComputation*. This map
  //     must contain all computations which the newly constructed instruction
  //     calls.
  static absl::StatusOr<std::unique_ptr<HloInstruction>> CreateFromProto(
      const HloInstructionProto& proto,
      const absl::flat_hash_map<int64_t, HloInstruction*>& instruction_map,
      const absl::flat_hash_map<int64_t, HloComputation*>& computation_map = {},
      bool prohibit_empty_literal = true);

  // Creates a parameter-retrieving instruction.
  static std::unique_ptr<HloInstruction> CreateParameter(
      int64_t parameter_number, const Shape& shape, absl::string_view name);

  // Creates a literal constant instruction.
  static std::unique_ptr<HloInstruction> CreateConstant(Literal literal);

  // Creates an Iota instruction.
  static std::unique_ptr<HloInstruction> CreateIota(const Shape& shape,
                                                    int64_t iota_dimension);

  // Creates a Top-K instruction returning the top k values along the last
  // dimension of the input operand.
  //
  // - `k` indicates how many elements to return in the last dimension.
  // - `largest` indicates whether to return the largest or smallest elements.
  static std::unique_ptr<HloInstruction> CreateTopK(const Shape& shape,
                                                    HloInstruction* input,
                                                    int64_t k, bool largest);

  // Creates a get tuple element instruction.
  static std::unique_ptr<HloInstruction> CreateGetTupleElement(
      const Shape& shape, HloInstruction* operand, int64_t index);

  // Creates a get tuple element instruction.
  static std::unique_ptr<HloInstruction> CreateGetTupleElement(
      HloInstruction* operand, int64_t index);

  // Creates a random number generation instruction that fills a shape with
  // random numbers from a given distribution.
  //
  // The parameters to the instruction are interpreted as follows:
  //
  //  - If `distribution` is RNG_UNIFORM, generates a number in range
  //    [param0, param1).
  //
  //  - If `distribution` is RNG_NORMAL, generates a normally-distributed value
  //    with mean `param0` and standard deviation `param1`.
  static std::unique_ptr<HloInstruction> CreateRng(
      const Shape& shape, RandomDistribution distribution,
      absl::Span<HloInstruction* const> parameters);

  // Creates a stateless random bit generator instruction that fills a shape
  // with random bits.
  static std::unique_ptr<HloInstruction> CreateRngBitGenerator(
      const Shape& shape, HloInstruction* state, RandomAlgorithm algorithm);

  // Creates an instruction to update the random number generator state to
  // reflect the new state after `delta` units of 32 random bits are generated
  // and returns the old state.
  static std::unique_ptr<HloInstruction> CreateRngGetAndUpdateState(
      const Shape& shape, int64_t delta);

  // Creates a unary instruction (one operand).
  // Precondition: opcode must be a legitimate unary operation.
  static std::unique_ptr<HloInstruction> CreateUnary(
      const Shape& shape, HloOpcode opcode, HloInstruction* operand,
      std::optional<ResultAccuracy> result_accuracy = std::nullopt);

  // Creates a binary instruction (two operands).
  // Precondition: opcode must be a legitimate binary operation.
  static std::unique_ptr<HloInstruction> CreateBinary(const Shape& shape,
                                                      HloOpcode opcode,
                                                      HloInstruction* lhs,
                                                      HloInstruction* rhs);

  // Creates a ternary instruction (three operands).
  // Precondition: opcode must be a legitimate ternary operation.
  static std::unique_ptr<HloInstruction> CreateTernary(const Shape& shape,
                                                       HloOpcode opcode,
                                                       HloInstruction* lhs,
                                                       HloInstruction* rhs,
                                                       HloInstruction* ehs);

  // Creates a variadic instruction (variable number of operands).
  // Precondition: opcode must be a legitimate variadic operation.
  static std::unique_ptr<HloInstruction> CreateVariadic(
      const Shape& shape, HloOpcode opcode,
      absl::Span<HloInstruction* const> operands);

  // Creates a map instruction, where the computation (given by the handle) is
  // applied element-wise to every element in operands (across the operands,
  // at a given index)
  static std::unique_ptr<HloInstruction> CreateMap(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* map_computation);

  // Creates a convolution op, where rhs is the convolutional filter
  // and window describes how the filter is applied to lhs.
  static std::unique_ptr<HloInstruction> CreateConvolve(
      const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
      int64_t feature_group_count, int64_t batch_group_count,
      const Window& window,
      const ConvolutionDimensionNumbers& dimension_numbers,
      const PrecisionConfig& precision_config);

  // Creates an FFT op, of the type indicated by fft_type.
  static std::unique_ptr<HloInstruction> CreateFft(
      const Shape& shape, HloInstruction* operand, FftType fft_type,
      absl::Span<const int64_t> fft_length);

  // Creates an asynchronous start, update, and done op.
  static std::unique_ptr<HloInstruction> CreateAsyncStart(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* async_computation,
      absl::string_view async_execution_thread = kMainExecutionThread);
  static std::unique_ptr<HloInstruction> CreateAsyncUpdate(
      const Shape& shape, HloInstruction* operand);
  static std::unique_ptr<HloInstruction> CreateAsyncDone(
      const Shape& shape, HloInstruction* operand);

  // Creates a copy-start op, indicating whether this is a cross-program
  // prefetch or not.
  static std::unique_ptr<HloInstruction> CreateCopyStart(
      const Shape& shape, HloInstruction* operand,
      std::optional<int> cross_program_prefetch_index = std::nullopt);

  // Creates a compare op, performing the comparison specified in direction.
  static std::unique_ptr<HloInstruction> CreateCompare(
      const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
      Comparison::Direction direction,
      std::optional<Comparison::Type> type = std::nullopt);

  static std::unique_ptr<HloInstruction> CreateTriangularSolve(
      const Shape& shape, HloInstruction* a, HloInstruction* b,
      const TriangularSolveOptions& options);

  static std::unique_ptr<HloInstruction> CreateCholesky(
      const Shape& shape, HloInstruction* a, const CholeskyOptions& options);

  // Creates a dot op with operands 'lhs' and 'rhs' with contracting and batch
  // dimensions specified in 'dimension_numbers'. If 'sparsity' is set, then
  // 'sparse_meta' must also be present (and have the same size).
  // Note: 'sparsity' argument is eventually moved in the HloDotInstruction
  // constructor, so no extra copies are created.
  static std::unique_ptr<HloInstruction> CreateDot(
      const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
      const DotDimensionNumbers& dimension_numbers,
      const PrecisionConfig& precision_config,
      std::vector<SparsityDescriptor> sparsity = {},
      absl::Span<HloInstruction* const> sparse_meta = {});

  // Creates a ragged dot op with operands 'lhs', 'rhs', and 'group_sizes', with
  // contracting, batch, ragged, and group dimensions specified in
  // 'dimension_numbers'.
  static std::unique_ptr<HloInstruction> CreateRaggedDot(
      const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
      HloInstruction* group_sizes,
      const RaggedDotDimensionNumbers& dimension_numbers,
      const PrecisionConfig& precision_config);

  // Creates a reduce-precision op, where operand is the data to reduce in
  // precision, and exponent_bits and mantissa_bits describe the precision to
  // reduce it to.
  static std::unique_ptr<HloInstruction> CreateReducePrecision(
      const Shape& shape, HloInstruction* operand, int exponent_bits,
      int mantissa_bits);

  // Creates an all-gather op, which concats the operands of all participants
  // along all_gather_dimension. The replica_groups, channel_id, and
  // use_global_device_ids arguments are identical to those in all-reduce,
  // except that the order of the group members determines the concatenation
  // order of inputs from different participants.
  static std::unique_ptr<HloInstruction> CreateAllGather(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      int64_t all_gather_dimension, const CollectiveDeviceList& device_list,
      bool constrain_layout, const std::optional<int64_t>& channel_id,
      bool use_global_device_ids);

  ABSL_DEPRECATED("Use CollectiveDeviceList instead of list of ReplicaGroup.")
  static std::unique_ptr<HloInstruction> CreateAllGather(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      int64_t all_gather_dimension,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids);

  // Creates an all-gather-start op, which concats the operands of all
  // participants
  // along all_gather_dimension. The replica_groups, channel_id, and
  // use_global_device_ids arguments are identical to those in all-reduce,
  // except that the order of the group members determines the concatenation
  // order of inputs from different participants. Needs to be used in
  // conjunction of a AllGatherDone op that synchronizes and returns the result.
  static std::unique_ptr<HloInstruction> CreateAllGatherStart(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      int64_t all_gather_dimension, const CollectiveDeviceList& device_list,
      bool constrain_layout, const std::optional<int64_t>& channel_id,
      bool use_global_device_ids);

  ABSL_DEPRECATED("Use CollectiveDeviceList instead of list of ReplicaGroup.")
  static std::unique_ptr<HloInstruction> CreateAllGatherStart(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      int64_t all_gather_dimension,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids);

  // Creates a cross replica reduction op.
  //
  // `reduction_computation`: the reduction function.
  //
  // `replica_groups`: each ReplicaGroup contains a list of replica id. If
  // empty, all replicas belong to one group in the order of 0 - (n-1).
  // Allreduce will be applied within subgroups.
  // For example, we have 4 replicas, then replica_groups={{0,2},{1,3}} means,
  // replica 0 and 2 are in subgroup 0, replica 1 and 3 are in subgroup 1.
  //
  // `channel_id`: for Allreduce nodes from different modules, if
  // they have the same channel_id, they will be 'Allreduce'd. If
  // empty, Allreduce will not be applied cross modules.
  static std::unique_ptr<HloInstruction> CreateAllReduce(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      const CollectiveDeviceList& device_list, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids);

  ABSL_DEPRECATED("Use CollectiveDeviceList instead of list of ReplicaGroup.")
  static std::unique_ptr<HloInstruction> CreateAllReduce(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids);

  // Creates a reduce-scatter operation which reduces its inputs across the
  // given replica groups and then scatters the reduced data across the N
  // participants.
  static std::unique_ptr<HloInstruction> CreateReduceScatter(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      const CollectiveDeviceList& device_list, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids,
      int64_t scatter_dimension);

  ABSL_DEPRECATED("Use CollectiveDeviceList instead of list of ReplicaGroup.")
  static std::unique_ptr<HloInstruction> CreateReduceScatter(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids,
      int64_t scatter_dimension);

  // Creates an asynchronous cross replica reduction op.
  //
  // `reduction_computation`: the reduction function.
  //
  // `replica_groups`: each ReplicaGroup contains a list of replica id. If
  // empty, all replicas belong to one group in the order of 0 - (n-1).
  // Allreduce will be applied within subgroups.
  // For example, we have 4 replicas, then replica_groups={{0,2},{1,3}} means,
  // replica 0 and 2 are in subgroup 0, replica 1 and 3 are in subgroup 1.
  //
  // `channel_id`: for Allreduce nodes from different modules, if
  // they have the same channel_id, they will be 'Allreduce'd. If
  // empty, Allreduce will not be applied cross modules.
  static std::unique_ptr<HloInstruction> CreateAllReduceStart(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      const CollectiveDeviceList& device_list, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids);

  ABSL_DEPRECATED("Use CollectiveDeviceList instead of list of ReplicaGroup.")
  static std::unique_ptr<HloInstruction> CreateAllReduceStart(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids);

  // An all-to-all op takes N array operands of the same shape and scatters them
  // to N replicas.  Each replica gathers the results into a tuple.
  //
  // For example, suppose we have 3 replicas, with replica i passing inputs
  // [a_i, b_i, c_i] to its all-to-all op.  Then the resulting tuples are
  //
  //   replica 0: (a_0, a_1, a_2)
  //   replica 1: (b_0, b_1, b_2)
  //   replica 2: (c_0, c_1, c_2).
  //
  // If replica_groups is set, the op is sharded and the replicas are permuted.
  // To explain by way of example, suppose we have replica_groups={{1,2},{3,0}}.
  // Then each replica passes two operands, say [a_i, b_i], and the result is
  //
  //   replica 0: (b_3, b_0)
  //   replica 1: (a_1, a_2)
  //   replica 2: (b_1, b_2)
  //   replica 3: (a_3, a_0).
  //
  // All replica groups must have the same number of elements, and the number of
  // operands must be equal to the size of one replica group.  Each replica must
  // appear in exactly one group.
  //
  // Note that in addition to supporting this instruction, XlaBuilder also
  // supports a higher-level instruction which takes one input and slices it,
  // performs AllToAll and then concatenates the results into a single array.
  static std::unique_ptr<HloInstruction> CreateAllToAll(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      const CollectiveDeviceList& device_list, bool constrain_layout,
      const std::optional<int64_t>& channel_id,
      const std::optional<int64_t>& split_dimension = std::nullopt);

  ABSL_DEPRECATED("Use CollectiveDeviceList instead of list of ReplicaGroup.")
  static std::unique_ptr<HloInstruction> CreateAllToAll(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id,
      const std::optional<int64_t>& split_dimension = std::nullopt);

  // The RaggedAllToAll instruction performs a collective all-to-all operation,
  // where the input and output are ragged tensors.
  //
  // Ragged tensors are defined by a set of three tensors:
  // *) ‘data’: the ‘data’ tensor is “ragged” along its outermost dimension,
  //   along which each indexed element has variable size.
  // *) ‘offsets’: the ‘offsets’ tensor indexes the outermost dimension of the
  //  ‘data’ tensor, and represents the starting offset of each ragged element
  //  of the ‘data’ tensor.
  // *) ‘sizes’: the ‘sizes’ tensor represents the size of each ragged element
  //  of the ‘data’ tensor, where the size is specified in units of
  //  sub-elements. A sub-element is defined as the suffix of the ‘data’ tensor
  //  shape obtained by removing the outermost “ragged” dimension.
  // *) The ‘offsets’ and ‘sizes’ tensors must have the same size.
  //
  // An example ragged tensor
  // data: [8,3] =
  //  {{a,b,c},{d,e,f},{g,h,i},{j,k,l},{m,n,o},{p,q,r},{s,t,u},{v,w,x}}
  // offsets: [3] = {0, 1, 4}
  // sizes: [3] = {1, 3, 4}
  //
  // Index 'data' at 'offsets'[0], 'sizes'[0]'
  // {a,b,c}
  //
  // Index 'data' at 'offsets'[1], 'sizes'[1]'
  // {d,e,f},{g,h,i},{j,k,l}
  //
  // Index 'data' at 'offsets'[2], 'sizes'[2]'
  // {m,n,o},{p,q,r},{s,t,u},{v,w,x}
  //
  //
  // ``output_offsets`` must be sharded in a way that each replica has offsets
  // in the target replica output perspective.
  //
  // For i-th output offset, the current replica will send
  // `input[input_offsets[i]:input_offsets[i]+input_sizes[i]]` update to
  // `i`-th replica that will be written to
  // `output_i[output_offsets[i]:output_offsets[i]+send_sizes[i]]` in `i`-th
  // replica ``output``.
  //
  // For example, if we have 2 replicas:
  //
  // replica 0:
  //   input: [1, 2, 2]
  //   output: [0, 0, 0, 0]
  //   input_offsets: [0, 1]
  //   send_sizes: [1, 2]
  //   output_offsets: [0, 0]
  //   recv_sizes: [1, 1]
  //
  // replica 1:
  //   input: [3, 4, 0]
  //   output: [0, 0, 0, 0]
  //   input_offsets: [0, 1]
  //   send_sizes: [1, 1]
  //   output_offsets: [1, 2]
  //   recv_sizes: [2, 1]
  //
  // replica 0's result will be: [1, 3, 0, 0]
  // replica 1's result will be: [2, 2, 4, 0]
  //
  // The ragged all-to-all HLO has the following arguments:
  //   input: ragged input data tensor.
  //   output: ragged output data tensor.
  //   input_offsets: ragged input offsets tensor.
  //   send_sizes: ragged send sizes tensor.
  //   output_offsets: array of ragged offsets in the target replica output.
  //   recv_sizes: ragged recv sizes tensor.
  //
  // The '*_offsets' and '*_sizes' tensors must all have the same shape.
  // Two shapes are supported for the '*_offsets' and '*_sizes' tensors:
  //   *) [num_devices] where ragged-all-to-all may send at most one update to
  //      each remote device in the replica group. For example:
  //
  //      for (remote_device_id : replica_group) {
  //        SEND input[input_offsets[remote_device_id]],
  //             output[output_offsets[remote_device_id]],
  //             send_sizes[remote_device_id]
  //      }
  //
  //   *) [num_devices, num_updates] where ragged-all-to-all may send up to
  //      'num_updates' updates the same remote device (each at different
  //      offsets), for each remote device in the replica group. For example:
  //
  //      for (remote_device_id : replica_group) {
  //        for (update_idx : num_updates) {
  //          SEND input[input_offsets[remote_device_id][update_idx]],
  //               output[output_offsets[remote_device_id][update_idx]]],
  //               send_sizes[remote_device_id][update_idx]
  //        }
  //      }
  //
  // The output buffer is passed in as an input (and aliased in the output),
  // to support incremental updates to the same buffer.
  //
  static std::unique_ptr<HloInstruction> CreateRaggedAllToAll(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      const CollectiveDeviceList& device_list,
      const std::optional<int64_t>& channel_id);

  ABSL_DEPRECATED("Use CollectiveDeviceList instead of list of ReplicaGroup.")
  static std::unique_ptr<HloInstruction> CreateRaggedAllToAll(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::Span<const ReplicaGroup> replica_groups,
      const std::optional<int64_t>& channel_id);

  // Creates a communication instruction that broadcasts data cross replicas.
  // Data is sent from to the first replica id in each group to the other ids in
  // the same group. If a replica id is not a in any replica group, the output
  // on that replica is a tensor consists of 0(s) in `shape`.
  static std::unique_ptr<HloInstruction> CreateCollectiveBroadcast(
      const Shape& shape, absl::Span<HloInstruction* const> operand,
      const CollectiveDeviceList& device_list, bool constrain_layout,
      const std::optional<int64_t>& channel_id);

  ABSL_DEPRECATED("Use CollectiveDeviceList instead of list of ReplicaGroup.")
  static std::unique_ptr<HloInstruction> CreateCollectiveBroadcast(
      const Shape& shape, absl::Span<HloInstruction* const> operand,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id);

  // Creates a communication instruction that permutes data cross replicas.
  // Data is sent/received according to the (source_replica_id,
  // target_replica_id) pairs in `source_target_pairs`. If a replica id is not a
  // target_replica_id in any pair, the output on that replica is a tensor
  // consists of 0(s) in `shape`.
  static std::unique_ptr<HloInstruction> CreateCollectivePermute(
      const Shape& shape, HloInstruction* operand,
      const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
      const std::optional<int64_t>& channel_id);
  static std::unique_ptr<HloInstruction> CreateCollectivePermute(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
      const std::optional<int64_t>& channel_id);

  static std::unique_ptr<HloInstruction> CreateCollectivePermute(
      const Shape& shape, HloInstruction* input, HloInstruction* output,
      HloInstruction* input_start_indices, HloInstruction* output_start_indices,
      absl::Span<const std::pair<int64_t, int64_t>> source_target_pairs,
      absl::Span<const std::vector<int64_t>> slice_sizes,
      const std::optional<int64_t>& channel_id);

  // Creates a communication instruction that initiates the start of
  // CollectivePermute.
  static std::unique_ptr<HloInstruction> CreateCollectivePermuteStart(
      const Shape& shape, HloInstruction* operand,
      const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
      const std::optional<int64_t>& channel_id);
  static std::unique_ptr<HloInstruction> CreateCollectivePermuteStart(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
      const std::optional<int64_t>& channel_id);

  static std::unique_ptr<HloInstruction> CreateCollectivePermuteStart(
      const Shape& shape, HloInstruction* input, HloInstruction* output,
      HloInstruction* input_start_indices, HloInstruction* output_start_indices,
      absl::Span<const std::pair<int64_t, int64_t>> source_target_pairs,
      absl::Span<const std::vector<int64_t>> slice_sizes,
      const std::optional<int64_t>& channel_id);

  // Creates an instruction that returns a U32 replica ID.
  static std::unique_ptr<HloInstruction> CreateReplicaId(
      const Shape& shape = ShapeUtil::MakeShape(U32, {}));

  // Creates an instruction that returns a U32 partition ID.
  static std::unique_ptr<HloInstruction> CreatePartitionId(
      const Shape& shape = ShapeUtil::MakeShape(U32, {}));

  // Creates a conversion instruction, where operand is the data to convert and
  // shape is the target shape for the conversion.
  static std::unique_ptr<HloInstruction> CreateConvert(const Shape& shape,
                                                       HloInstruction* operand);

  // Creates a bitcast instruction, where operand is the data to
  // convert and shape is the target shape for the conversion.
  static std::unique_ptr<HloInstruction> CreateBitcast(const Shape& shape,
                                                       HloInstruction* operand);

  // Creates a bitcast conversion instruction, where operand is the data to
  // convert and shape is the target shape for the conversion.
  static std::unique_ptr<HloInstruction> CreateBitcastConvert(
      const Shape& shape, HloInstruction* operand);

  // Creates a stochastic conversion instruction, where operand is the data to
  // convert, random is a given random input to determine the rounding direction
  // and shape is the target shape for the conversion.
  static std::unique_ptr<HloInstruction> CreateStochasticConvert(
      const Shape& shape, HloInstruction* operand, HloInstruction* random);

  // Creates an infeed instruction, which reads data of the given shape from the
  // Infeed interface of the device. infeed_shape is the shape of the data
  // received from the infeed *not* the shape of the infeed instruction which
  // is a tuple containing the infeed_shape and the TOKEN.
  static std::unique_ptr<HloInstruction> CreateInfeed(
      const Shape& infeed_shape, HloInstruction* token_operand,
      const std::string& config);

  // Creates an outfeed instruction, which outputs data. outfeed_shape is the
  // shape of the data being outfed *not* the shape of the outfeed instruction
  // which is a TOKEN.
  static std::unique_ptr<HloInstruction> CreateOutfeed(
      const Shape& outfeed_shape, HloInstruction* operand,
      HloInstruction* token_operand, absl::string_view outfeed_config);

  // Creates an asynchronous send instruction with the given channel id, which
  // initiates sending the operand data to a unique receive instruction in
  // another computation that has the same channel id. If is_host_transfer is
  // true, then this Send operation transfers data to the host.
  static std::unique_ptr<HloInstruction> CreateSend(
      HloInstruction* operand, HloInstruction* token,
      std::optional<int64_t> channel_id, bool is_host_transfer);

  // Blocks until data transfer for the Send instruction (operand) is complete.
  // The operand must be kSend.
  static std::unique_ptr<HloInstruction> CreateSendDone(
      HloInstruction* operand, std::optional<int64_t> channel_id,
      bool is_host_transfer);

  // Creates an asynchronous receive instruction with the given channel id,
  // which allocates resources to receive data of the given shape from a unique
  // send instruction in another computation that has the same channel id.  If
  // is_host_transfer is true, then this Recv operation transfers data from the
  // host.
  static std::unique_ptr<HloInstruction> CreateRecv(
      const Shape& shape, HloInstruction* token,
      std::optional<int64_t> channel_id, bool is_host_transfer);

  // Blocks until data transfer for the Recv instruction (operand) is complete
  // and returns the receive buffer. The operand must be kRecv.
  static std::unique_ptr<HloInstruction> CreateRecvDone(
      HloInstruction* operand, std::optional<int64_t> channel_id,
      bool is_host_transfer);

  // Creates a slice instruction, where the operand is sliced by the given
  // start/limit indices.
  static std::unique_ptr<HloInstruction> CreateSlice(
      const Shape& shape, HloInstruction* operand,
      absl::Span<const int64_t> start_indices,
      absl::Span<const int64_t> limit_indices,
      absl::Span<const int64_t> strides);

  // Creates a slice instruction, where the first operand is sliced by
  // start indices specified in the second operand, and by size specified in
  // 'slice_sizes'.
  static std::unique_ptr<HloInstruction> CreateDynamicSlice(
      const Shape& shape, HloInstruction* operand,
      absl::Span<HloInstruction* const> start_indices,
      absl::Span<const int64_t> slice_sizes);

  // Creates a dynamic update slice instruction, which updates a slice
  // of 'operand' with 'update' and 'start_indices'.
  static std::unique_ptr<HloInstruction> CreateDynamicUpdateSlice(
      const Shape& shape, HloInstruction* operand, HloInstruction* update,
      absl::Span<HloInstruction* const> start_indices);

  // Creates a concatenate instruction, where the operands are concatenated on
  // the provided dimension.
  static std::unique_ptr<HloInstruction> CreateConcatenate(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      int64_t dimension);

  // Creates a reduce instruction, where the computation (given by the handle)
  // is applied successively to every element in operand. For example, let f be
  // the function to apply, which takes 2 arguments, an accumulator and the
  // current value. Let init be an initial value (which is normally chosen to be
  // the identity element for f, e.g. 0 if f is addition).
  // Then the reduce HLO will compute:
  // f(f(init, value0), value1), ...)
  static std::unique_ptr<HloInstruction> CreateReduce(
      const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
      absl::Span<const int64_t> dimensions_to_reduce,
      HloComputation* reduce_computation);

  // A more general, multiple-argument version of the above.
  // The function to apply, f, now takes N arguments:
  // [accumulator0, accumulator1, ..., accumulatorN, value0, value1, ...,
  // init_valueN], and returns an N-tuple. The performed computation is (for
  // commutative and associative f operators) equivalent to:
  //
  // f_1 = f(init0, ...  initN, input0.value0, ..., inputN.value0)
  // f_2 = f(f_1.tuple_element(0), ..., f_1.tuple_element(N), input0.value1,
  // ..., inputN.value1)
  // ...
  static std::unique_ptr<HloInstruction> CreateReduce(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::Span<HloInstruction* const> init_values,
      absl::Span<const int64_t> dimensions_to_reduce,
      HloComputation* reduce_computation);

  // Helper version where the operands are given by a single instruction which
  // either is a tuple of size `init_values`, or a single input, in which case
  // size of `init_values` is one.
  static std::unique_ptr<HloInstruction> CreateReduce(
      const Shape& shape, HloInstruction* tuple_of_instructions,
      absl::Span<HloInstruction* const> init_values,
      absl::Span<const int64_t> dimensions_to_reduce,
      HloComputation* reduce_computation);

  // Creates a reduce-window instruction, where the computation (given
  // by the handle) is applied window-wise at each valid window
  // position in the operand.
  static std::unique_ptr<HloInstruction> CreateReduceWindow(
      const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
      const Window& window, HloComputation* reduce_computation);

  // A more general, multiple-argument version of the above.
  // The reduce_computation being applied,now takes N arguments:
  // [accumulator0, accumulator1, ..., accumulatorN, value0, value1, ...,
  // valueN], and returns an N-tuple. The operands and init_values now each
  // contain a span of N input arrays and n initial values.
  static std::unique_ptr<HloInstruction> CreateReduceWindow(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::Span<HloInstruction* const> init_values, const Window& window,
      HloComputation* reduce_computation);

  // Creates a batch-norm-training instruction.
  static std::unique_ptr<HloInstruction> CreateBatchNormTraining(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* offset, float epsilon, int64_t feature_index);

  // Creates a batch-norm-inference instruction.
  static std::unique_ptr<HloInstruction> CreateBatchNormInference(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* offset, HloInstruction* mean, HloInstruction* variance,
      float epsilon, int64_t feature_index);

  // Creates a batch-norm-grad instruction.
  static std::unique_ptr<HloInstruction> CreateBatchNormGrad(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* mean, HloInstruction* variance,
      HloInstruction* grad_output, float epsilon, int64_t feature_index);

  // Creates a scatter computation that scatters the `source` array to the
  // selected indices of each window.
  static std::unique_ptr<HloInstruction> CreateSelectAndScatter(
      const Shape& shape, HloInstruction* operand, HloComputation* select,
      const Window& window, HloInstruction* source, HloInstruction* init_value,
      HloComputation* scatter);

  // Creates a broadcast instruction.
  static std::unique_ptr<HloInstruction> CreateBroadcast(
      const Shape& shape, HloInstruction* operand,
      absl::Span<const int64_t> broadcast_dimensions);

  // Creates a sequence of instructions that performs an explicit broadcast of
  // the operand to the target shape.
  //
  // Interior HLOs are passed to "adder", but the "root" HLO of the sequence is
  // returned as a unique_ptr for API consistency with other factory methods in
  // this interface.
  //
  // TODO(b/72173833) Ideally HloComputations would always be present, and so
  // the adder being passed by the caller would not be necessary.
  static std::unique_ptr<HloInstruction> CreateBroadcastSequence(
      const Shape& output_shape, HloInstruction* operand,
      absl::FunctionRef<HloInstruction*(std::unique_ptr<HloInstruction>)>
          adder);

  // Creates a pad instruction, where the operand is padded on the edges and
  // between the elements with the given padding value.
  static std::unique_ptr<HloInstruction> CreatePad(
      const Shape& shape, HloInstruction* operand,
      HloInstruction* padding_value, const PaddingConfig& padding_config);

  // Creates a reshape instruction, where the operand is flattened row-major
  // order and then reshaped to the given result shape.
  static std::unique_ptr<HloInstruction> CreateReshape(
      const Shape& shape, HloInstruction* operand,
      int64_t inferred_dimension = -1);

  // Creates a dynamic reshape instruction. Similar to reshape but dynamic
  // dimensions sizes are provided as additional variadic arguments.
  //
  // Precondition: dim_sizes.size() == shape.dimensions_size()
  static std::unique_ptr<HloInstruction> CreateDynamicReshape(
      const Shape& shape, HloInstruction* data_operand,
      absl::Span<HloInstruction* const> dim_sizes);

  // Creates a transpose instruction which permutes the operand dimensions.
  static std::unique_ptr<HloInstruction> CreateTranspose(
      const Shape& shape, HloInstruction* operand,
      absl::Span<const int64_t> dimensions);

  // Creates a n-ary sort op with a 'compare' computation which is used for
  // comparisons in the sorting algorithm. 'compare' gets 2 * n parameters,
  // where parameters 2 * i and 2 * i + 1 are the values of the i-th operand at
  // specific index positions which should be compared, and should return a
  // PRED. 'is_stable' specifies whether stable sorting is required.
  static std::unique_ptr<HloInstruction> CreateSort(
      const Shape& shape, int64_t dimension,
      absl::Span<HloInstruction* const> operands, HloComputation* compare,
      bool is_stable);

  // Creates a while instruction, given a condition computation, a body
  // computation, and the initial value for the input of the computations. For
  // example, shape: S32, condition: i -> i < 1000, body: i -> i * 2, init: 1
  // corresponds to the C code below.
  // int32_t i = 1; int32_t result = while(i < 1000) { i = i * 2 }
  static std::unique_ptr<HloInstruction> CreateWhile(const Shape& shape,
                                                     HloComputation* condition,
                                                     HloComputation* body,
                                                     HloInstruction* init);

  static std::unique_ptr<HloInstruction> CreateConditional(
      const Shape& shape, HloInstruction* pred,
      HloInstruction* true_computation_arg, HloComputation* true_computation,
      HloInstruction* false_computation_arg, HloComputation* false_computation);

  static std::unique_ptr<HloInstruction> CreateConditional(
      const Shape& shape, HloInstruction* branch_index,
      absl::Span<HloComputation* const> branch_computations,
      absl::Span<HloInstruction* const> branch_computation_args);

  static std::unique_ptr<HloInstruction> CreateGather(
      const Shape& shape, HloInstruction* operand,
      HloInstruction* start_indices,
      const GatherDimensionNumbers& gather_dim_numbers,
      absl::Span<const int64_t> slice_sizes, bool indices_are_sorted);

  static std::unique_ptr<HloInstruction> CreateScatter(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloInstruction* scatter_indices,
      absl::Span<HloInstruction* const> updates,
      HloComputation* update_computation,
      const ScatterDimensionNumbers& scatter_dim_numbers,
      bool indices_are_sorted, bool unique_indices);

  static std::unique_ptr<HloInstruction> CreateScatter(
      const Shape& shape, HloInstruction* operand,
      HloInstruction* scatter_indices, HloInstruction* updates,
      HloComputation* update_computation,
      const ScatterDimensionNumbers& scatter_dim_numbers,
      bool indices_are_sorted, bool unique_indices);

  // Creates a kDomain instruction which delimits an HLO domain which have
  // the provided user and operand side metadata.
  static std::unique_ptr<HloInstruction> CreateDomain(
      const Shape& shape, HloInstruction* operand,
      std::unique_ptr<DomainMetadata> operand_side_metadata,
      std::unique_ptr<DomainMetadata> user_side_metadata);

  // Creates a fusion instruction. A fusion instruction contains one or more
  // fused instructions forming an expression with a single root
  // "fused_root". Additional instructions can be added to the fusion
  // instruction with the method FuseInstruction.
  static std::unique_ptr<HloInstruction> CreateFusion(
      const Shape& shape, FusionKind fusion_kind, HloInstruction* fused_root,
      absl::string_view prefix = "");

  static std::unique_ptr<HloInstruction> CreateFusion(
      const Shape& shape, FusionKind fusion_kind,
      absl::Span<HloInstruction* const> operands,
      HloComputation* fusion_computation, absl::string_view prefix = "");

  // Creates a call instruction that applies the given computation on the given
  // operands. "shape" is the resultant shape.
  static std::unique_ptr<HloInstruction> CreateCall(
      const Shape& shape, HloInstruction* called_computation_root);

  static std::unique_ptr<HloInstruction> CreateCall(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* computation);

  // Creates a composite call instruction that applies the given computation on
  // the given operands. "shape" is the resultant shape.
  static std::unique_ptr<HloInstruction> CreateCompositeCall(
      const Shape& shape, HloInstruction* decomposition_root,
      const std::string& name, const std::string& attributes, int64_t version);

  static std::unique_ptr<HloInstruction> CreateCompositeCall(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* decomposition, const std::string& name,
      const std::string& attributes, int64_t version);

  // Creates a custom call instruction that applies the given custom call target
  // to the given operands. "opaque" can be an arbitrary string with a
  // backend-specific interpretation. "shape" is the resultant shape.
  static std::unique_ptr<HloInstruction> CreateCustomCall(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::string_view custom_call_target, std::string opaque = "",
      CustomCallApiVersion api_version = API_VERSION_ORIGINAL);

  // Overload with a to_apply computation.
  static std::unique_ptr<HloInstruction> CreateCustomCall(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* to_apply, absl::string_view custom_call_target,
      std::string opaque = "",
      CustomCallApiVersion api_version = API_VERSION_ORIGINAL);

  // Overload with multiple computations. The called computations can have
  // different function signatures.
  static std::unique_ptr<HloInstruction> CreateCustomCall(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::Span<HloComputation* const> called_computations,
      absl::string_view custom_call_target, std::string opaque = "",
      CustomCallApiVersion api_version = API_VERSION_ORIGINAL);

  // Overload which constrains the layouts of the operand and result. 'shape'
  // and 'operand_shapes_with_layout' must have layouts.
  // 'operand_shapes_with_layout' must have a compatible element for each
  // operand.
  static std::unique_ptr<HloInstruction> CreateCustomCall(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::string_view custom_call_target,
      absl::Span<const Shape> operand_shapes_with_layout,
      std::string opaque = "",
      CustomCallApiVersion api_version = API_VERSION_ORIGINAL);

  // Creates a tuple instruction with the given elements. This is a convenience
  // wrapper around CreateVariadic.
  static std::unique_ptr<HloInstruction> CreateTuple(
      absl::Span<HloInstruction* const> elements);

  // Creates a reverse instruction, which reverses the order of the elements
  // in the specified dimensions.
  static std::unique_ptr<HloInstruction> CreateReverse(
      const Shape& shape, HloInstruction* operand,
      absl::Span<const int64_t> dimensions);

  // Creates a Afterall instruction used for joining or creating new values of
  // token type which thread through side-effecting operations. Operands must
  // all be tokens, calls without operands generates a token.
  static std::unique_ptr<HloInstruction> CreateAfterAll(
      absl::Span<HloInstruction* const> operands);

  // Creates an AfterAll instruction which creates a token type out of thin air
  // (no operands). This is a separate method from CreateAfterAll to facility
  // the removal of operand-less AfterAll instructions.
  // TODO(b/110532604): Remove this capability of creating a token from nothing
  // when we plumb a primordial token from the entry computation.
  static std::unique_ptr<HloInstruction> CreateToken();

  static std::unique_ptr<HloInstruction> CreateGetDimensionSize(
      const Shape& shape, HloInstruction* operand, int64_t dimension);

  static std::unique_ptr<HloInstruction> CreateSetDimensionSize(
      const Shape& shape, HloInstruction* operand, HloInstruction* val,
      int64_t dimension);

  static std::unique_ptr<HloInstruction> CreateAddDependency(
      HloInstruction* data_operand, HloInstruction* token_operand);

  // Returns true if `execution_thread` is included in the
  // `execution_threads_set`.
  static bool IsThreadIncluded(
      absl::string_view execution_thread,
      const absl::flat_hash_set<absl::string_view>& execution_threads_set);

  // Returns the opcode for this instruction.
  HloOpcode opcode() const { return opcode_; }
  HloOpcode* mutable_opcode() { return &opcode_; }

  // Returns whether this instruction is the root of its parent computation.
  bool IsRoot() const { return is_root_; }
  void MarkAsRoot() { is_root_ = true; }
  void MarkAsNonRoot() { is_root_ = false; }

  // Does this instruction have no users.
  bool IsDead() const { return users_.empty() && !IsRoot(); }

  // Returns true if this instruction has a side effect, irrespective of whether
  // any called computations may contain an instruction with side effects.
  bool HasSideEffectNoRecurse() const;

  // Returns true if this instruction has a side effect. An instruction has a
  // side effect if it uses certain opcodes or calls a computation with a side
  // effect.
  virtual bool HasSideEffect() const;

  // Returns the result shape of this instruction.
  const Shape& shape() const;

  // Returns the (mutable) result shape of this instruction.
  Shape* mutable_shape() { return &shape_; }

  // Returns the ith operand to this instruction.
  const HloInstruction* operand(int64_t i) const;

  // Returns the ith operand to this instruction.
  HloInstruction* mutable_operand(int64_t i);

  // Returns the number of operands to this instruction.
  int64_t operand_count() const { return operands_.size(); }

  // Returns the vector of operands of this instruction.
  using InstructionVector = absl::InlinedVector<HloInstruction*, 2>;
  const InstructionVector& operands() const { return operands_; }
  InstructionVector mutable_operands() { return operands_; }

  // Returns the vector of unique operands, in the same order they are found
  // within the operand vector.
  InstructionVector unique_operands() const;

  // Returns the first index of 'target' that occurs in the operands sequence.
  // Precondition: target must be an operand (or a fatal error will occur).
  int64_t operand_index(const HloInstruction* target) const;

  // Returns all indices of 'target' that occur in the operands sequence.
  // Precondition: target must be an operand (or a fatal error will occur).
  std::vector<int64_t> operand_indices(const HloInstruction* target) const;

  // Returns the number of users of this instruction.
  int64_t user_count() const { return users_.size(); }

  // Returns the users of this instruction.
  const PtrVec<HloInstruction*>& users() const { return users_.vec(); }

  // Returns the index of the user in the users() vector.
  //
  // Precondition: `user` is a user of the instruction.
  int64_t UserId(HloInstruction* user) { return users_.UserId(user); }

  // Returns true if this instruction is a user of 'instruction'.
  bool IsUserOf(const HloInstruction* instruction) const {
    return instruction->users_.Contains(this);
  }

  // Adds a control dependency from this instruction to the given
  // instruction. This instruction becomes a control predecessor of
  // 'instruction', and 'instruction' becomes a control successor of this
  // instruction. Returns an error status if either of the given instructions
  // does not belong to the same computation.
  //
  // This is used to enforce an additional ordering requirement that is not
  // captured by normal data dependencies, such as ordering among Send or Recv
  // operations to avoid deadlock.
  absl::Status AddControlDependencyTo(HloInstruction* instruction);

  // Removes a previously added control dependency from this instruction to
  // 'instruction'.
  absl::Status RemoveControlDependencyTo(HloInstruction* instruction);

  // Drops all control predecessors and successors from this HLO instruction.
  absl::Status DropAllControlDeps();

  // Drops all control predecessors and successors from this HLO instruction,
  // and the maintain the transitivie control dependencies between
  // control predecessors and control successors.
  absl::Status SafelyDropAllControlDependencies();

  // Returns if instruction has any control dependencies.
  bool HasControlDependencies() const;

  // Copies the control predecessors and successors on this HLO instruction to
  // `inst`.  Does not do a deep copy so this makes sense only if `inst` and
  // this HLO are in the same module.
  //
  // Depending on the use cases we see in practice, in the future we may
  // consider folding the logic here into Clone, CloneWithNewOperands and
  // ReplaceAllUsesWith by treating control dependencies like data dependencies.
  absl::Status CopyAllControlDepsFrom(const HloInstruction* inst) {
    return inst->CopyAllControlDepsTo(this, this);
  }

  // Copies all control dependencies of this instruction to start/end. Copies
  // all control predecessors of this instruction to control predecessors of
  // `start` and copies all control successors of this instruction to control
  // successors of `end`.
  absl::Status CopyAllControlDepsTo(HloInstruction* start,
                                    HloInstruction* end) const;

  // Returns the set of control predecessors (successors) of this
  // instruction. Control predecessors (successors) must execute before (after)
  // the current instruction.
  const PtrVec<HloInstruction*>& control_predecessors() const {
    return rare()->control_predecessors;
  }
  const PtrVec<HloInstruction*>& control_successors() const {
    return rare()->control_successors;
  }

  // Returns true if 'other' performs the same computation as this instruction.
  bool Identical(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloInstruction*, const HloInstruction*)>
          eq_operands = std::equal_to<const HloInstruction*>(),
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations = std::equal_to<const HloComputation*>(),
      bool layout_sensitive = true, bool sharding_sensitive = false) const {
    return IdenticalInternal(other, eq_operands, eq_computations,
                             layout_sensitive, sharding_sensitive,
                             /*ignore_channel_id_values=*/false,
                             /*ignore_commutative_operand_order=*/false);
  }
  // Returns true if 'other' is the same kind of op as this instruction. For
  // regular ops, it just checks whether the opcode is the same, for ops like
  // e.g. kCompare, it also checks extra attributes.
  bool SameOp(const HloInstruction& other) const {
    return opcode() == other.opcode() &&
           IdenticalSlowPath(other, std::equal_to<const HloComputation*>());
  }

  // Same as Identical() but ignores the order of commutative operands (e.g.
  // considers add(a,b) equal to add(b,a)).
  bool IdenticalIgnoringCommutativeOperandOrder(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloInstruction*, const HloInstruction*)>
          eq_operands = std::equal_to<const HloInstruction*>(),
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations = std::equal_to<const HloComputation*>(),
      bool layout_sensitive = true, bool sharding_sensitive = false) const {
    return IdenticalInternal(other, eq_operands, eq_computations,
                             layout_sensitive, sharding_sensitive,
                             /*ignore_channel_id_values=*/false,
                             /*ignore_commutative_operand_order=*/true);
  }

  // Same as Identical() but ignores channel ID value mismatches, as long as
  // both have channel IDs or neither has a channel ID.
  bool IdenticalIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloInstruction*, const HloInstruction*)>
          eq_operands = std::equal_to<const HloInstruction*>(),
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations = std::equal_to<const HloComputation*>(),
      bool layout_sensitive = true, bool sharding_sensitive = false) const {
    return IdenticalInternal(other, eq_operands, eq_computations,
                             layout_sensitive, sharding_sensitive,
                             /*ignore_channel_id_values=*/true,
                             /*ignore_commutative_operand_order=*/true);
  }

  // Allow subclasses to contribute additional attributes to the hash.
  virtual void HashAdditionalAttributes(absl::HashState h) const {};

  // Generates a hash value of an HLO instruction. Hash considers
  // information on opcode, shape, number of operands, and other relevant
  // additional attributes (e.g. literal values, parameters, etc.).
  template <typename H>
  friend H AbslHashValue(H h, const HloInstruction& hlo) {
    h = H::combine(std::move(h), hlo.opcode(), hlo.shape());
    if (!hlo.IsCrossModuleAllReduce()) {
      h = H::combine(std::move(h), hlo.operand_count());
    }
    // Allow subclasses to mix additional data into h before returning
    hlo.HashAdditionalAttributes(absl::HashState::Create(&h));
    return h;
  }

  // Returns whether the instruction has a constant operand.
  bool HasConstantOperand() const;

  // Replaces the use of this instruction in "user" with "new_producer". Note
  // that there might be multiple uses of this instruction in "user"; all will
  // be replaced.
  //
  // If user is a fusion instruction, this function will remove any duplicated
  // operands of it which could be created due to this replacement.
  absl::Status ReplaceUseWith(HloInstruction* user,
                              HloInstruction* new_producer);

  // Same as ReplaceUseWith(), but new_producer can have a different shape.
  absl::Status ReplaceUseWithDifferentShape(HloInstruction* user,
                                            HloInstruction* new_producer);

  // Same as ReplaceUseWith but only replaces the use at the given operand
  // number.
  absl::Status ReplaceUseWith(HloInstruction* user, int operand_number,
                              HloInstruction* new_producer);
  absl::Status ReplaceUseWithDifferentShape(HloInstruction* user,
                                            int operand_number,
                                            HloInstruction* new_producer);

  // Replaces the specified operand with new_operand. The old and new operands
  // must have compatible shapes ignoring floating-point precision.
  //
  // This function does NOT remove duplicated operands even if this instruction
  // is a fusion, so that the existing operand numbers do not change.
  absl::Status ReplaceOperandWith(int64_t operand_num,
                                  HloInstruction* new_operand);

  // Same as ReplaceOperandWith(), but new_operand can have a different shape.
  absl::Status ReplaceOperandWithDifferentShape(int64_t operand_num,
                                                HloInstruction* new_operand);

  // Decomposes fusion back to individual parts.
  absl::Status Defuse();

  // Unfuses the given instruction from its fusion computation. If the given
  // instruction is not fused, this is a no-op and returns nullptr. Returns a
  // pointer to the newly unfused instruction if successful. Currently, fused
  // instructions with parameter or constant operands are supported.
  absl::StatusOr<HloInstruction*> UnfuseInstruction(
      HloInstruction* instruction);

  // Replaces all uses of this instruction with the new producer. If
  // new_producer is a user of this instruction then new_producer remains a use
  // of this instruction to avoid introducing cycles into the graph.
  //
  // If this instruction is the root of its computation, sets the computation's
  // root to new_producer.
  //
  // The new producer must have a compatible shape ignoring floating-point
  // precision.
  //
  // If a user is a fusion instruction, this function will remove any duplicated
  // operands of it which could be created due to this replacement.
  //
  // trigger is a string used in the error message if the new and the
  // current instruction don't have a compatible shape.
  absl::Status ReplaceAllUsesWith(HloInstruction* new_producer,
                                  absl::string_view trigger = "");

  // Same as ReplaceAllUsesWith, but new_producer can have a different shape.
  absl::Status ReplaceAllUsesWithDifferentShape(HloInstruction* new_producer);

  // Same as ReplaceAllUsesWith, but only replace given set of users.
  absl::Status ReplaceUsesWith(absl::Span<HloInstruction* const> users,
                               HloInstruction* new_producer);
  absl::Status ReplaceAllUsesWithDifferentShape(
      absl::Span<HloInstruction* const> users, HloInstruction* new_producer);

  // Performs a postorder DFS visit using this node as the root. If
  // call_finish_visit is true, then DfsHloVisitor::FinishVisit is called when
  // complete. If ignore_control_predecessors is true, instructions only
  // reachable via control dependencies will not be visited, and the postorder
  // will not take control dependencies into account. It is as if the control
  // dependencies didn't exist in the graph at all. If cross_computation is
  // true, DFS will go across the computation boundary (i.e., from an
  // instruction to the root instruction of a computation it calls).
  template <typename HloInstructionPtr>
  absl::Status Accept(DfsHloVisitorBase<HloInstructionPtr>* visitor,
                      bool call_finish_visit = true,
                      bool ignore_control_predecessors = false,
                      bool cross_computation = false);
  absl::Status Accept(ConstDfsHloVisitor* visitor,
                      bool call_finish_visit = true,
                      bool ignore_control_predecessors = false,
                      bool cross_computation = false) const {
    return const_cast<HloInstruction*>(this)->Accept(
        visitor, call_finish_visit, ignore_control_predecessors,
        cross_computation);
  }

  // Same as Accept() above, but the order of operand and control predecessor
  // visitation is determined by the given operand order; if compare(A, B) ==
  // true, A is visited before B.
  using CompareFunction =
      absl::FunctionRef<bool(const HloInstruction*, const HloInstruction*)>;
  absl::Status AcceptWithOperandOrder(DfsHloVisitor* visitor,
                                      CompareFunction operand_order,
                                      bool call_finish_visit = true);

  // Visit this instruction and only this instruction with the given visitor.
  template <typename HloInstructionPtr>
  absl::Status Visit(DfsHloVisitorBase<HloInstructionPtr>* visitor);
  absl::Status Visit(ConstDfsHloVisitor* visitor) const {
    return const_cast<HloInstruction*>(this)->Visit(visitor);
  }

  // Returns the first non-GetTupleElement ancestor instruction of 'hlo'.
  // If the first non-GTE ancestor is tuple-shaped, populates 'index' with the
  // (possibly nested) tuple indices used on the path from ancestor to 'hlo'.
  std::pair<const HloInstruction*, ShapeIndex> LatestNonGteAncestorAndIndex()
      const;

  std::pair<HloInstruction*, ShapeIndex> LatestNonGteAncestorAndIndex() {
    auto rv =
        const_cast<const HloInstruction*>(this)->LatestNonGteAncestorAndIndex();
    return {const_cast<HloInstruction*>(rv.first), rv.second};
  }

  // Same as LatestNonGteAncestorAndIndex, but just returns the HloInstruction.
  const HloInstruction* LatestNonGteAncestor() const;

  HloInstruction* LatestNonGteAncestor() {
    return const_cast<HloInstruction*>(
        const_cast<const HloInstruction*>(this)->LatestNonGteAncestor());
  }

  // Returns true whether this instruction is effectively a bitcast. Currently,
  // this means it either is a bitcast, or it is a transpose that is effectively
  // a bitcast.
  bool IsEffectiveBitcast() const;

  // Returns true if this instruction is asynchronous with the
  // async_execution_thread set to `execution_thread`.
  bool IsAsyncInstructionWithExecutionThread(
      absl::string_view execution_thread) const {
    return IsAsynchronous() && async_execution_thread() == execution_thread;
  };

  // Gets/sets the to_apply HloComputation for Call, Map, Reduce, etc.
  // The setter should only be called by HloModule or HloComputation methods.
  //
  // Precondition: The instruction has a valid to_apply_ field.
  HloComputation* to_apply() const;
  void set_to_apply(HloComputation* computation);
  // Whether the instruction has a valid to_apply_ field.
  bool has_to_apply() const;

  // Gets/sets the while_condition or while_body HloComputation for While. The
  // setters should only be called by HloModule or HloComputation methods.
  //
  // Precondition: The instruction is a While instruction.
  HloComputation* while_condition() const;
  HloComputation* while_body() const;
  void set_while_condition(HloComputation* computation);
  void set_while_body(HloComputation* computation);

  HloInstruction* while_init() const;

  // Gets/sets the true and false HloComputation for Conditional.
  //
  // Precondition: The instruction is a predicated Conditional instruction.
  HloComputation* true_computation() const;
  HloComputation* false_computation() const;

  // Gets the branch HloComputations for Conditional.
  //
  // Precondition: The instruction is a Conditional instruction.
  const PtrVec<HloComputation*>& branch_computations() const;
  int32_t branch_count() const;
  HloComputation* branch_computation(int32_t b) const;
  int32_t branch_index(HloComputation* computation) const;
  // Sets a branch HloComputation for Conditional.
  // The setter should only be called by HloModule or HloComputation methods.
  //
  // Precondition: The instruction is a Conditional instruction.
  void set_branch_computation(int b, HloComputation* computation);

  // Returns a string for the signature of this instruction if considered as a
  // function, e.g. the signature of an F32 add is (F32, F32) -> F32.
  std::string SignatureString() const;

  // Prints a debugging string that represents this instruction.
  void Print(Printer* printer) const {
    return Print(printer, HloPrintOptions::Default());
  }
  void Print(Printer* printer, const HloPrintOptions& options) const;

  // Returns a debugging string that represents this instruction.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  //
  // TODO(b/73348663): Make ToString() adaptive to the size of the string by
  // default, backing off on providing full information for very large strings,
  // or provide a different name for a ToString-like function that does that.
  std::string ToString() const;
  std::string ToString(const HloPrintOptions& options) const;

  // Components of the Print() and ToString() representation:

  // Helper class for PrintExtraAttributes.
  class AttributePrinter {
   public:
    explicit AttributePrinter(std::function<Printer*()> next_printer)
        : next_printer_(std::move(next_printer)) {}

    void Next(absl::FunctionRef<void(Printer*)> print_func) {
      print_func(next_printer_());
    }

   private:
    std::function<Printer*()> next_printer_;
  };
  // Prints the string representation of op-specific attributes.
  void PrintExtraAttributes(AttributePrinter& printer,
                            const HloPrintOptions& options) const;

  // Returns string representation of op-specific attributes.
  std::vector<std::string> ExtraAttributesToString(
      const HloPrintOptions& options) const;

  // As ToString, but returns a shorter string.
  std::string ToShortString() const;

  // Prints an instruction to a string.
  //
  // The canonical string representation needs to name operands and instruction
  // names in a consistent way. This is implemented through the
  // canonical_name_map.
  void PrintWithCanonicalNameMap(Printer* printer,
                                 const HloPrintOptions& options,
                                 CanonicalNameMap* canonical_name_map) const;

  // Returns a serialized representation of this instruction.
  virtual HloInstructionProto ToProto() const;

  // Returns a category for the HLO. This could be something like "convolution"
  // or "elementwise".
  virtual std::string ToCategory() const;

  // Returns true if this instruction is fused, ie contained within a fusion
  // instruction.
  bool IsFused() const;

  bool IsLoopFusion() const;
  bool IsInputFusion() const;
  bool IsOutputFusion() const;
  bool IsCustomFusion() const;

  // Returns true if this instruction can be legally fused into a fusion
  // instruction.
  bool IsFusible() const;

  bool IsCustomCall(absl::string_view target) const;
  bool IsCustomCall(absl::Span<const absl::string_view> targets) const;

  // Returns the sharding applied to this operator.
  // REQUIRES: has_sharding() is true.
  const HloSharding& sharding() const {
    CHECK(has_sharding());
    return *sharding_;
  }
  std::shared_ptr<const HloSharding> sharding_ptr() const { return sharding_; }

  // Returns the sharding applied to this operator, or default_ if none exists.
  const HloSharding& sharding_or_default(const HloSharding& default_) const {
    return sharding_ ? *sharding_ : default_;
  }
  // Returns the sharding unique device, if any.
  std::optional<int64_t> sharding_unique_device() const {
    if (sharding_ == nullptr) {
      return std::nullopt;
    }
    return sharding_->UniqueDevice();
  }
  // Sets the sharding of this operator. Should only be called by HloModule or
  // HloComputation methods.
  void set_sharding(HloSharding sharding) {
    set_sharding(std::make_shared<HloSharding>(std::move(sharding)));
  }
  void set_sharding(std::shared_ptr<const HloSharding> sharding) {
    sharding_ = std::move(sharding);
  }
  // Copies the sharding of another instruction, this is more efficient than
  // set_sharding(hlo->sharding()) because it avoids a deep copy and shares the
  // storage. Note that if the other instruction has no sharding set, it also
  // clears the sharding of the current instruction.
  void copy_sharding(const HloInstruction* hlo) {
    set_sharding(hlo->sharding_ptr());
  }
  void set_single_sharding(const HloSharding& sharding);
  // Sets a sharding that assigns the current instruction to device.
  void set_device_sharding(int64_t device) {
    set_single_sharding(HloSharding::AssignDevice(device));
  }
  // Remove any sharding from this operator.
  void clear_sharding() { sharding_ = nullptr; }
  // Return true if this operator has a sharding assigned.
  bool has_sharding() const { return sharding_ != nullptr; }
  // Checks whether the instruction has compatible sharding with the other
  // instruction.
  bool has_compatible_sharding(const HloInstruction* other) const {
    if (!has_sharding()) {
      return !other->has_sharding();
    }
    return other->has_sharding() ? sharding() == other->sharding() : false;
  }

  // When creating a new instruction which either replaces, or shifts up (kCopy
  // insertion case), another instruction, we need to make sure the certain
  // properties of the new instruction are copied into the derived one. As of
  // today, the metadata and sharding will be propagated to the derived
  // instruction.
  void SetupDerivedInstruction(HloInstruction* derived_instruction) const;

  // Clones the HLO instruction. The clone will have the same opcode, shape, and
  // operands. After creation the clone has no uses. "this" (the instruction
  // cloned from) is not changed. Suffix is the string to append to the name of
  // the instruction to form the name of the cloned instruction.
  // Ignores the control predecessors and successors of this HLO instruction.
  std::unique_ptr<HloInstruction> Clone(
      const std::string& suffix = "clone",
      HloCloneContext* context = nullptr) const;

  // Clones the HLO instruction as above but with new shape.
  std::unique_ptr<HloInstruction> CloneWithNewShape(
      const Shape& shape, const std::string& suffix = "clone",
      HloCloneContext* context = nullptr) const;

  // Clones the HLO instruction as above but with new shape and operands.
  std::unique_ptr<HloInstruction> CloneWithNewOperands(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context = nullptr) const;

  // Clones the HLO instruction with new shape, operands and suffix.
  std::unique_ptr<HloInstruction> CloneWithNewOperands(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      const std::string& suffix, HloCloneContext* context = nullptr) const;

  // Implementation for non-common logic of CloneWithNewOperands.
  // CloneWithNewOperands forwards to this method for some of the intstruction
  // types.
  virtual std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const {
    // TODO(b/80131774): This should be pure virtual.
    LOG(FATAL) << "Unimplemented method.";
  }

  // Returns the computations this instruction directly calls (if any).
  const PtrVec<HloComputation*>& called_computations() const {
    return rare()->called_computations;
  }
  bool has_called_computations() const {
    return has_rare() && !called_computations().empty();
  }

  // Returns true iff an instruction of type "opcode" might have non-empty
  // called_computations.
  static bool MightHaveCalledComputations(HloOpcode opcode);

  // Replaces all called computations based on a map function. This is needed
  // when we clone hlo_computations and want to let the instructions to point
  // to the newly cloned nodes.
  void ReplaceCalledComputations(
      absl::FunctionRef<HloComputation*(HloComputation*)> map_function);

  // Clears out the called computations.
  //
  // This is, in particular, necessary when inlining function bodies into their
  // caller. If there were side-effecting operations in the called computations,
  // the call itself is considered side-effecting and thus cannot be removed. By
  // clearing out the computations, we reflect the fact that all side-effecting
  // properties have been reflected in the caller, and make the call HLO
  // removable.
  virtual void ClearCalledComputations();

  // Returns true if this instruction performs an elementwise operation on
  // `operand_idx`-th operand. An instruction is elementwise on an operand iff,
  // to compute the output at index {i_0,i_1,...,i_n}, the only element required
  // from the operand (if any) is the element at {i_0,i_1,...,i_n}.
  //
  // Note on performance: when this instruction is kFusion, this method, in the
  // worst case, scans all fused instructions. We could speed this up by
  // caching.
  bool IsElementwiseOnOperand(int64_t operand_idx) const;

  // Returns true if this instruction is elementwise on all its operands.
  bool IsElementwise() const;

  static bool IsOpElementwise(HloOpcode opcode);

  // Returns true if this is a cross module all-reduce instruction.
  bool IsCrossModuleAllReduce() const;

  // Returns true if this is a cross-replica all-reduce instruction.
  bool IsCrossReplicaAllReduce() const;

  // Returns true if this instruction is binary and elementwise.
  bool IsElementwiseBinary() const;

  // Returns whether this instruction may reuse elements of its `i`th operand.
  bool ReusesOperandElements(int64_t i) const;

  // Returns the indices that the given operand appear in the operand list of
  // this instruction. Note that an instruction can use the same operand
  // multiple times.
  absl::InlinedVector<int64_t, 4> OperandIndices(
      const HloInstruction* operand) const;

  // Convenience helper for ShapeUtil::InsertedOrDeleted1SizedDimensions. If
  // this reshape merely inserts or deletes 1-sized dimensions, return the input
  // indices of the deleted dimensions and the output indices of the inserted
  // dimensions.
  //
  // Precondition: this op must be a reshape.
  std::optional<ShapeUtil::ShapeEqualityDescriptor>
  ReshapeMerelyInsertsOrDeletes1SizedDimensions() const;

  // Gets the string identifier for this instruction.
  absl::string_view name() const { return name_; }

  // Sets the string identifier for this instruction. Name will be sanitized to
  // match the regexp "[a-zA-Z_][a-zA-Z0-9_.-]*".
  //
  // See also HloModule::SetAndUniquifyInstrName(), which does this plus
  // UniquifyName().
  void SetAndSanitizeName(absl::string_view name) {
    name_ = NameUniquer::GetSanitizedName(name);
  }

  // Use the given NameUniquer to select a unique name for the instruction based
  // on the instruction's existing name.
  //
  // See also HloModule::SetAndUniquifyInstrName(), which does this plus
  // SetAndSanitizeName().
  void UniquifyName(NameUniquer* name_uniquer);

  // Use the `module`'s name uniquer to select a unique name for this
  // instruction based on the instruction's existing name.
  void UniquifyName(HloModule* module);

  // Use the `module`s `NewUniqueInstructionId` to set the id of this
  // instruction.
  void UniquifyId(HloModule* module);

  // Clear the unique ID of the instruction so that it can be re-assigned, such
  // as for the purpose of compacting the instruction unique IDs.
  void ClearUniqueIdInternal() { unique_id_ = -1; }

  // Set the unique id for this instruction to "id"
  void SetUniqueId(int id) {
    CHECK_EQ(unique_id_, -1);  // Should not be assigned already
    CHECK_GE(id, 0);
    unique_id_ = id;
  }

  // Return the unique ID assigned to this node via SetUniqueId (or -1
  // if no id has been assigned yet).
  int unique_id() const { return unique_id_; }

  bool has_backend_config() const { return !backend_config_.empty(); }

  void clear_backend_config() { backend_config_ = BackendConfigWrapper(); }

  void CopyBackendConfigFrom(const HloInstruction* other) {
    backend_config_ = BackendConfigWrapper(other->backend_config_);
  }

  // Replaces the frontend attributes with the provided argument.
  void set_frontend_attributes(FrontendAttributes frontend_attributes) {
    if (!has_rare() && frontend_attributes.map().empty()) {
      return;
    }
    mutable_rare()->frontend_attributes = std::move(frontend_attributes);
  }

  // Adds attributes only if they not already present in the HloInstruction.
  // Skips all atributes already present in the HloInstruction.
  void add_frontend_attributes(FrontendAttributes frontend_attributes) {
    if (!frontend_attributes.map().empty()) {
      mutable_rare()->frontend_attributes.mutable_map()->insert(
          frontend_attributes.map().begin(), frontend_attributes.map().end());
    }
  }

  // Adds a single attribute only if it not already present in the
  // HloInstruction. Returns false if the attribute was already present.
  bool add_frontend_attribute(const std::string& key,
                              const std::string& value) {
    auto it =
        mutable_rare()->frontend_attributes.mutable_map()->insert({key, value});
    return it.second;
  }

  size_t erase_frontend_attribute(const std::string& key) {
    return mutable_rare()->frontend_attributes.mutable_map()->erase(key);
  }

  // Adds or overrides a single attribute in the HloInstruction.
  void set_frontend_attribute(const std::string& key,
                              const std::string& value) {
    (*mutable_rare()->frontend_attributes.mutable_map())[key] = value;
  }

  bool has_frontend_attributes() const {
    return has_rare() && !rare()->frontend_attributes.map().empty();
  }

  const FrontendAttributes& frontend_attributes() const {
    return rare()->frontend_attributes;
  }

  std::optional<std::string> get_frontend_attribute(
      const std::string& key) const {
    auto it = rare()->frontend_attributes.map().find(key);
    if (it == rare()->frontend_attributes.map().end()) {
      return std::nullopt;
    }
    return it->second;
  }

  void set_is_composite(bool is_composite) {
    if (!has_rare() && !is_composite) {
      return;
    }
    mutable_rare()->is_composite = is_composite;
  }

  // Return the is_composite attribute. This attribute is only relevant for
  // kCall instructions used as a Composite op.
  bool is_composite() const { return has_rare() && rare()->is_composite; }

  const ResultAccuracy& result_accuracy() const {
    return rare()->result_accuracy;
  }

  bool has_result_accuracy() const {
    return has_rare() && (result_accuracy().has_tolerance() ||
                          result_accuracy().mode() != ResultAccuracy::DEFAULT);
  }

  bool equal_result_accuracy(const HloInstruction* other) const {
    return result_accuracy().has_tolerance() ==
               other->result_accuracy().has_tolerance() &&
           result_accuracy().tolerance().atol() ==
               other->result_accuracy().tolerance().atol() &&
           result_accuracy().tolerance().rtol() ==
               other->result_accuracy().tolerance().rtol() &&
           result_accuracy().tolerance().ulps() ==
               other->result_accuracy().tolerance().ulps() &&
           result_accuracy().mode() == other->result_accuracy().mode();
  }

  void add_single_statistic(Statistic statistic) {
    *mutable_rare()->statistics_viz.add_statistics() = std::move(statistic);
  }

  void set_stat_index_to_visualize(int64_t index) {
    mutable_rare()->statistics_viz.set_stat_index_to_visualize(index);
  }

  // Whether this specific instruction has statistics
  bool has_statistics() const { return !statistics_viz().statistics().empty(); }

  // Whether any instruction within the same HLO module as this has statistics
  bool module_has_statistics() const {
    return statistics_viz().stat_index_to_visualize() == -1;
  }

  const Statistic& statistic_to_visualize() const {
    return statistics_viz().statistics().at(
        statistics_viz().stat_index_to_visualize());
  }

  void set_statistics_viz(StatisticsViz statistics_viz) {
    mutable_rare()->statistics_viz = std::move(statistics_viz);
  }

  const StatisticsViz& statistics_viz() const { return rare()->statistics_viz; }

  template <typename T>
  using EnableIfProto = typename std::enable_if_t<
      std::is_base_of<tsl::protobuf::Message, T>::value>;

  // Returns the backend-specific configuration for how a backend should compile
  // this HLO. The meaning of the field is backend specific. Not for use before
  // or during general HLO optimization, since HLO optimizations do not preserve
  // this field and they cannot interpret it due to its meaning being backend
  // specific. Except for CustomCall, where this field is preserved and no
  // general HLO optimization needs to interpret it.
  //
  // ConfigProto should be a protobuf Message type.
  template <typename ConfigProto, EnableIfProto<ConfigProto>* = nullptr>
  absl::StatusOr<ConfigProto> backend_config() const {
    ConfigProto proto;
    TF_RETURN_IF_ERROR(backend_config_.GetProto(&proto));
    return proto;
  }

  absl::Status set_backend_config(const tsl::protobuf::Message& proto) {
    backend_config_ = BackendConfigWrapper(proto);
    return absl::OkStatus();
  }

  // Getter/setter for raw JSON-encoded backend config.  Prefer the
  // functions above that deal in proto Messages where possible.
  const std::string& raw_backend_config_string() const {
    return backend_config_.GetRawString();
  }
  void set_raw_backend_config_string(std::string config_str) {
    backend_config_ = BackendConfigWrapper(std::move(config_str));
  }

  bool is_default_config() const { return is_default_config_; }
  void set_default_config() { is_default_config_ = true; }

  // Returns the information used to tell the implementation information about
  // what sort of precision is requested. The meaning of the field is backend
  // specific. At the moment, it is only supported for kConvolution and kDot.
  // Transformations on one kDot or kConvolution to another will preserve this
  // information. Transformations to other HLOs will not preserve this
  // information but it is presumed that the alternate lowering is strictly
  // superior.
  // Precondition: opcode must be kConvolution or kDot.
  const PrecisionConfig& precision_config() const;
  PrecisionConfig* mutable_precision_config();

  // Sets the result accuracy for this instruction. Supported for unary ops
  // with multiple implementations.
  void set_result_accuracy(ResultAccuracy result_accuracy) {
    mutable_rare()->result_accuracy = std::move(result_accuracy);
  }

  // Sets the debug metadata for this instruction, excluding creation_pass_id,
  // which should never be copied anywhere.
  void set_metadata(const OpMetadata& metadata) { *metadata_ = metadata; }

  void set_size_of_generated_code_in_bytes(int64_t code_size_in_bytes) {
    metadata_->set_size_of_generated_code_in_bytes(code_size_in_bytes);
  }
  void set_size_of_memory_working_set_in_bytes(
      int64_t working_set_size_in_bytes) {
    metadata_->set_size_of_memory_working_set_in_bytes(
        working_set_size_in_bytes);
  }
  void set_metadata_op_name(const std::string& name) {
    metadata_->set_op_name(name);
  }
  void set_metadata_deduplicated_name(std::string deduplicated_name) {
    metadata_->set_deduplicated_name(std::move(deduplicated_name));
  }
  void set_metadata_scheduling_name(absl::string_view name) {
    metadata_->set_scheduling_name(std::string(name));
  }
  const OpMetadata& metadata() const { return *metadata_; }

  // Get the computation containing this instruction.
  const HloComputation* parent() const { return parent_; }
  HloComputation* parent() { return parent_; }

  // Returns the module for this instruction.
  HloModule* GetModule() const;

  // A method that sorts users_, control_predecessors_, and control_successors_
  // according to the orders used in sorted_instruction. The sorting is used
  // during cloning, to make clone behavior match uncloned behavior.
  void SortInstructionUsersAndControlLists(
      const MappedPtrContainerSorter<HloInstruction>::MapPtrFn& map_fn,
      const HloInstruction& sorted_instruction);

  // Old methods kept for smooth subclassing transition BEGIN.
  // NOTE: Refrain from adding more delegates, prefer down casting to subclasses
  // rather than using these methods.
  // TODO(b/80131774): Remove this code.

  // Delegates to HloBatchNormInstruction::feature_index.
  int64_t feature_index() const;

  // Delegates to HloBatchNormInstruction::epsilon.
  float epsilon() const;

  // Delegates to HloFftInstruction::fft_type.
  FftType fft_type() const;

  // Delegates to HloFftInstruction::fft_length.
  const std::vector<int64_t>& fft_length() const;

  // Delegates to HloChannelInstruction::channel_id.
  std::optional<int64_t> channel_id() const;
  void set_channel_id(const std::optional<int64_t>& channel_id);

  // Returns the dimension sizes or numbers associated with this instruction.
  virtual absl::Span<const int64_t> dimensions() const {
    LOG(FATAL) << "Unimplemented method.";
  }

  int64_t dimensions(int64_t index) const { return dimensions()[index]; }

  virtual std::vector<int64_t>* mutable_dimensions() {
    LOG(FATAL) << "Unimplemented method.";
  }

  // Delegates to HloConcatenateInstruction::concatenate_dimension.
  virtual int64_t concatenate_dimension() const;

  // Delegates to HloGetDimensionSizeInstruction::dimension.
  int64_t dimension() const;

  // Delegates to HloReshapeInstruction::inferred_dimension.
  int64_t inferred_dimension() const;

  // Returns whether this instruction does a rank-2 transposition.
  bool IsRank2Transpose() const;

  // Delegates to HloSliceInstruction::slice_start.
  int64_t slice_starts(int64_t dimension) const;
  const std::vector<int64_t>& slice_starts() const;
  std::vector<int64_t>* mutable_slice_starts();

  // Delegates to HloSliceInstruction::slice_limits.
  int64_t slice_limits(int64_t dimension) const;
  const std::vector<int64_t>& slice_limits() const;
  std::vector<int64_t>* mutable_slice_limits();

  // Delegates to HloSliceInstruction::slice_strides.
  int64_t slice_strides(int64_t dimension) const;
  const std::vector<int64_t>& slice_strides() const;
  std::vector<int64_t>* mutable_slice_strides();

  // Returns the literal associated with this instruction.
  const Literal& literal() const;

  // Returns whether the instruction is a constant.
  bool IsConstant() const;

  // Delegate to HloConstantInstruction::RelayoutConstant.
  void RelayoutConstant(const Layout& new_layout,
                        const ShapeIndex& shape_index = {});

  // Delegates to
  // HloCallableInstruction::AppendInstructionIntoCalledComputation.
  HloInstruction* AppendInstructionIntoCalledComputation(
      HloInstruction* instruction_to_append, bool add_output = false);

  // Delegates to HloFusionInstruction::AddFusionOperand.
  HloInstruction* AddFusionOperand(HloInstruction* new_operand);

  // Delegates to HloFusionInstruction::MergeFusionInstruction.
  void MergeFusionInstruction(HloInstruction* instruction_to_merge);

  // Delegates to HloFusionInstruction::MergeFusionInstructionIntoMultiOutput.
  void MergeFusionInstructionIntoMultiOutput(
      HloInstruction* instruction_to_merge);

  // Delegates to HloFusionInstruction::FuseInstruction.
  HloInstruction* FuseInstruction(HloInstruction* instruction_to_fuse);

  // Delegates to HloFusionInstruction::FuseInstructionIntoMultiOutput.
  HloInstruction* FuseInstructionIntoMultiOutput(
      HloInstruction* instruction_to_fuse);

  // Delegates to HloFusionInstruction::fused_instruction.
  HloComputation* fused_instructions_computation() const;

  // Delegates to HloFusionInstruction::fused_expression_root.
  HloInstruction* fused_expression_root() const;

  // Delegates to HloFusionInstruction::fused_instructions.
  tsl::gtl::iterator_range<HloInstructionUnwrappingConstIterator>
  fused_instructions() const;

  tsl::gtl::iterator_range<HloInstructionUnwrappingIterator>
  fused_instructions();

  // Delegates to HloFusionInstruction::fused_instruction_count.
  int64_t fused_instruction_count() const;

  // Delegates to HloFusionInstruction::fused_parameter.
  HloInstruction* fused_parameter(int64_t parameter_number) const;

  // Delegates to HloFusionInstruction::fused_parameters.
  const InstructionVector& fused_parameters() const;

  // Returns true if this instruction is a fusion instruction that generates
  // multiple outputs.
  bool IsMultiOutputFusion() const;

  // Delegates to HloFusionInstruction::fusion_kind.
  FusionKind fusion_kind() const;

  // Delegates to HloFusionInstruction::set_fusion_kind.
  void set_fusion_kind(FusionKind kind);

  // Delegates to HloRngInstruction::random_distribution.
  RandomDistribution random_distribution() const;

  // Delegates to HloParameterInstruction::parameter_number.
  int64_t parameter_number() const;

  // Delegates to
  // HloParameterInstruction::set_parameter_replicated_at_leaf_buffers.
  void set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool> parameter_replicated_at_leaf_buffers);
  void set_parameter_replicated_at_leaf_buffers(
      const std::vector<bool>& parameter_replicated_at_leaf_buffers);

  // Delegates to HloParameterInstruction::parameter_replicated_at_leaf_buffers.
  const std::optional<std::vector<bool>>& parameter_replicated_at_leaf_buffers()
      const;

  // Delegates to HloGetTupleElementInstruction::tuple_index.
  int64_t tuple_index() const;

  // Delegates to HloGetTupleElementInstruction::set_tuple_index.
  void set_tuple_index(int64_t new_tuple_index);

  // Delegates to HloReducePrecisionInstruction::exponent_bits.
  int32_t exponent_bits() const;

  // Delegates to HloReducePrecisionInstruction::mantissa_bits.
  int32_t mantissa_bits() const;

  // Delegates to HloInfeedInstruction::infeed_config.
  std::string infeed_config() const;

  // Delegates to HloInfeedInstruction::set_infeed_config.
  void set_infeed_config(const std::string& config);

  // Returns the config for the Outfeed instruction.
  const std::string& outfeed_config() const;

  // Delegates to HloOutfeedInstruction::set_outfeed_config.
  void set_outfeed_config(const std::string& config);

  // Returns the shape for the Outfeed instruction.
  const Shape& outfeed_shape() const;

  // Returns the mutable shape for the Outfeed instruction.
  Shape* mutable_outfeed_shape();

  // Delegates to HloCollectiveInstruction::replica_groups.
  // TODO(b/316622399): Remove usages of this method and replace with
  // device_list()->replica_groups().
  const std::vector<ReplicaGroup>& replica_groups() const;

  // Delegates to HloCollectiveInstruction::device_list.
  const CollectiveDeviceList& device_list() const;

  // Delegates to HloCollectivePermuteInstruction::source_target_pairs.
  const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs() const;

  // Returns data on the window in a windowed operation such as
  // convolution.
  virtual const Window& window() const {
    LOG(FATAL) << "Unimplemented method.";
  }

  // Sets the window data in a windowed operation such as convolution.
  virtual void set_window(const Window& window) {
    LOG(FATAL) << "Unimplemented method.";
  }

  // Returns the unique_indices field.
  virtual bool unique_indices() const { LOG(FATAL) << "Unimplemented method."; }

  // Returns data on the dimension numbers used for a convolution operation,
  // which may be a kConvolution instruction or a kCustomCall that implements a
  // convolution.
  const ConvolutionDimensionNumbers& convolution_dimension_numbers() const;

  // Sets the convolution dimension numbers on this instruction.  In general you
  // shouldn't need to call this; instead, specify the convolution dimension
  // numbers when you create the instruction.
  void set_convolution_dimension_numbers(
      const ConvolutionDimensionNumbers& dnums);

  // The number of feature groups. Must be a divisor of the input feature
  // dimension and output feature dimension.
  int64_t feature_group_count() const;

  void set_feature_group_count(int64_t feature_group_count);

  // The number of batch groups. Must be a divisor of the input batch dimension
  int64_t batch_group_count() const;

  void set_batch_group_count(int64_t batch_group_count);

  // Delegates to HloSelectAndScatterInstruction::select.
  HloComputation* select() const;

  // Delegates to HloSelectAndScatterInstruction::scatter.
  HloComputation* scatter() const;

  // Delegates to HloSelectAndScatterInstruction::set_select.
  void set_select(HloComputation* computation);

  // Delegates to HloSelectAndScatterInstruction::set_scatter.
  void set_scatter(HloComputation* computation);

  // Delegates to HloCustomCallInstruction::custom_call_target.
  const std::string& custom_call_target() const;
  void set_custom_call_target(absl::string_view target);

  // Delegates to HloPadInstruction::padding_config.
  const PaddingConfig& padding_config() const;
  PaddingConfig* mutable_padding_config();

  // Delegates to HloConvolutionInstruction::padding_type.
  PaddingType padding_type() const;

  // Delegates to HloDynamicSliceInstruction::slice_sizes.
  int64_t slice_sizes(int64_t dimension) const;

  // Delegates to HloDynamicSliceInstruction::dynamic_slice_sizes.
  const std::vector<int64_t>& dynamic_slice_sizes() const;

  // Delegates to HloCollectivePermuteInstruction::dynamic_slice_sizes.
  const std::vector<std::vector<int64_t>>& dynamic_slice_sizes_list() const;

  // Delegates to HloGatherInstruction::gather_dimension_numbers.
  const GatherDimensionNumbers& gather_dimension_numbers() const;
  // Delegates to HloGatherInstruction::gather_slice_sizes.
  absl::Span<const int64_t> gather_slice_sizes() const;

  // Delegates to HloScatterInstruction::scatter_dimension_numbers().
  const ScatterDimensionNumbers& scatter_dimension_numbers() const;

  // Delegates to HloDotInstruction::dot_dimension_numbers().
  const DotDimensionNumbers& dot_dimension_numbers() const;

  // Delegates to HloRaggedDotInstruction::ragged_dot_dimension_numbers().
  const RaggedDotDimensionNumbers& ragged_dot_dimension_numbers() const;

  // Delegates to HloDomainInstruction::operand_side_metadata().
  const DomainMetadata& operand_side_metadata() const;

  // Delegates to HloDomainInstruction::user_side_metadata().
  const DomainMetadata& user_side_metadata() const;

  // Returns true if the instruction is an async-start, async-update, or
  // async-done.
  bool IsAsynchronous() const;

  // Delagates to HloAsyncInstruction::async_chain_start().
  HloInstruction* async_chain_start() const;

  // Delagates to HloAsyncInstruction::async_done().
  HloInstruction* async_chain_done() const;

  // Returns the computation that will executed asynchronously.
  HloComputation* async_wrapped_computation() const;

  // Delagates to HloAsyncInstruction::async_wrapped_instruction().
  HloInstruction* async_wrapped_instruction() const;

  // Delagates to HloAsyncInstruction::async_wrapped_opcode().
  HloOpcode async_wrapped_opcode() const;

  // Delegates to HloAsyncInstruction::async_execution_thread().
  absl::string_view async_execution_thread() const;

  // Delegates to HloAsyncInstruction::set_async_execution_thread().
  void set_async_execution_thread(absl::string_view async_execution_thread);

  // Delegates to
  // HloCallableInstruction::RecursivelySetComputationsThreadName().
  void set_called_computations_execution_thread(
      absl::string_view async_execution_thread,
      bool skip_async_execution_thread_overwrite);

  // Delegates to HloCopyStartInstruction::is_cross_program_prefetch_index().
  std::optional<int> cross_program_prefetch_index() const;

  // Delegates to HloCompareInstruction::direction().
  ComparisonDirection comparison_direction() const;
  // Delegates to HloCompareInstruction::order().
  ComparisonOrder comparison_order() const;

  // Delegates to HloTriangularSolveInstruction::triangular_solve_options().
  const TriangularSolveOptions& triangular_solve_options() const;

  // Delegates to HloCholeskyInstruction::cholesky_options().
  const CholeskyOptions& cholesky_options() const;

  // Delegates to HloCallableInstruction::output_to_operand_aliasing().
  const std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>&
  output_operand_aliasing() const;

  // Delegates to HloCallableInstruction::set_output_to_operand_aliasing().
  void set_output_to_operand_aliasing(
      std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          aliasing);

  // Appends operand(s) to the list of operands and adds this instruction as a
  // user of the operand.
  void AppendOperand(HloInstruction* operand);
  void AppendOperands(absl::Span<HloInstruction* const> operands);

  // Old methods kept for smooth subclassing transition END.

  HloInstruction(const HloInstruction&) = delete;
  HloInstruction& operator=(const HloInstruction&) = delete;

  std::shared_ptr<OriginalValue> original_value() const;
  void set_original_value(std::shared_ptr<OriginalValue> original_value);

 protected:
  // Internal constructor for a given opcode/shape, other fields must be filled
  // by factory methods.
  HloInstruction(HloOpcode opcode, const Shape& shape);

  void RemoveAllOperands() { operands_.clear(); }

  void RemoveOperandAt(int index) {
    operands_.erase(operands_.begin() + index);
  }

  // Removes a list of operands with the given indices in ascending order.
  void RemoveOperandsAtAscendingIndices(
      absl::Span<const int> ascending_indices);

  void AppendComputation(HloComputation* computation);

  void DetachFrom(HloInstruction* usee) { usee->RemoveUser(this); }

  void set_called_computation(int index, HloComputation* computation);

  // Indices of computations in called_computations for instructions which call
  // multiple computations.
  enum {
    // kWhile computations.
    kBodyComputationIndex = 0,
    kConditionComputationIndex = 1,

    // kSelectAndScatter computations.
    kSelectComputationIndex = 0,
    kScatterComputationIndex = 1,

    // kConditional computations.
    kTrueComputationIndex = 0,
    kFalseComputationIndex = 1,
  };

  // Change instruction's name to have a given suffix.
  void AddSuffixToInstructionName(const absl::string_view suffix);

 private:
  friend class HloComputation;

  bool IdenticalInternal(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloInstruction*, const HloInstruction*)>
          eq_operands,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations,
      bool layout_sensitive, bool sharding_sensitive,
      bool ignore_channel_id_values,
      bool ignore_commutative_operand_order) const;

  // Set the computation containing this instruction.
  void set_parent(HloComputation* computation) { parent_ = computation; }

  // Implementation for non-common logic of PrintExtraAttributes.
  virtual void PrintExtraAttributesImpl(AttributePrinter& printer,
                                        const HloPrintOptions& options) const {}

  // Implementation for IsElementwise if operand_idx is nullopt and for
  // IsElementwiseOnOperand if otherwise.
  //
  // NOTE: For all instructions other than kFusion, being elementwise on one of
  // the operands is equivalent to being elementwise on all the operands.
  virtual bool IsElementwiseImpl(
      const std::optional<int64_t>& operand_idx) const;

  // Prints an operand to a string. Accessed by friend class HloInstruction.
  virtual void PrintOperandsWithCanonicalNameMap(
      Printer* printer, const HloPrintOptions& options,
      CanonicalNameMap* canonical_name_map) const;

  // See comments on Identical().
  virtual bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const;

  // Creates an n-ary elementwise operation.
  static std::unique_ptr<HloInstruction> CreateNary(
      const Shape& shape, HloOpcode opcode,
      absl::Span<HloInstruction* const> operands);

  // Adds a user for this instruction.
  void AddUser(HloInstruction* user) { users_.AddUser(user); }

  // Removes a user for this instruction.
  void RemoveUser(HloInstruction* user) { users_.RemoveUser(user); }

  // Helper for implementing backend_config().  Parses backend_config_ into the
  // given proto.
  absl::Status GetBackendConfigInternal(tsl::protobuf::Message* proto) const;

  // Mark this instruction as dead. Accessed by friend class HloInstruction.
  void MarkAsDead() { marked_as_dead_ = true; }

  // Has this instruction been marked as dead? Accessed by friend class
  // HloInstruction.
  bool IsMarkedAsDead() const { return marked_as_dead_; }

  // Rare is allocated lazily, only when any of its constituent fields are
  // non-empty.  This reduces the memory footprint of HloInstruction objects.
  struct Rare {
    // Computations called by this instruction.
    PtrVec<HloComputation*> called_computations;

    // The set of control predecessors of this instruction.
    // Note that the order of the instructions in the vector influences the
    // order computed in HloComputation::ComputeInstructionPostOrder, which may
    // influence the result of the compilation by changing the scheduling. We
    // are not sure if it matters.
    PtrVec<HloInstruction*> control_predecessors;

    // The set of control successors of this instruction.
    PtrVec<HloInstruction*> control_successors;

    // Attributes passed from the frontend to give hints to the backend about
    // how to compile this HLO.
    // HLO -> HLO transforms are expected to preserve these attributes on a
    // "best effort" basis only.
    // For example:
    //    x = const(10, frontend_attributes={x}
    //    y = const(10, frontend_attributes={y}
    //    z = add(x,y), frontend_attributes={y}
    // Could be simplified to:
    //    z' = const(20), frontend_attributes={?}
    FrontendAttributes frontend_attributes;

    // Used by kCall to determine if the Call instruction is a composite.
    bool is_composite;

    // Used to render an HLO graph when tracking the propagation desired values
    // through it.
    StatisticsViz statistics_viz;

    // Used to select different implementations for unary functions.
    ResultAccuracy result_accuracy;
  };

  static const Rare* const kEmptyRare;

  bool has_rare() const { return rare_ != nullptr; }

  // Return the allocated rare state, or the pointer to the static empty rare
  // state
  const Rare* rare() const {
    Rare* r = rare_.get();
    return (r == nullptr) ? kEmptyRare : r;
  }

  // Lazily allocate the Rare struct
  Rare* mutable_rare() {
    if (rare_ == nullptr) {
      rare_ = std::make_unique<Rare>();
    }
    return rare_.get();
  }

  // Users holds the list of users of an HloInstruction, plus it provides a fast
  // way for checking for presence of a potential user.
  class Users {
   public:
    Users() = default;
    ~Users();

    // No copying allowed
    Users(const Users&) = delete;
    Users& operator=(const Users&) = delete;

    bool empty() const { return users_.empty(); }
    int64_t size() const { return users_.size(); }
    const PtrVec<HloInstruction*>& vec() const { return users_; }

    void Clear();
    bool Contains(const HloInstruction* instruction) const;
    void AddUser(HloInstruction* user);
    void MaybeRemoveUser(HloInstruction* user);  // Remove user if present
    void RemoveUser(HloInstruction* user);       // REQUIRES: Contains(user)
    int64_t UserId(HloInstruction* user);
    void SortInstructionUsers(
        const MappedPtrContainerSorter<HloInstruction>::MapPtrFn& map_fn,
        const Users& sorted_instruction_users);
    bool CheckInvariants();

   private:
    void RebuildMap();

    PtrVec<HloInstruction*> users_;

    // If users_ is big, we also maintain a copy of the elements of users_
    // in a hash map to enable fast membership tests. The value in the map
    // contains the index of the instruction in the vector what enables fast
    // removal.
    static constexpr size_t kMapThreshold = 16;
    std::unique_ptr<absl::flat_hash_map<const HloInstruction*, int64_t>>
        user_map_;
  };

  int unique_id_;  // Unique to this HloInstruction within a HloModule
  uint32_t index_in_parent_;  // Index that identifies inst in HloComputation

  // Opcode for this instruction.
  HloOpcode opcode_;

  // This field is assigned to true when backend_config_ is assigned to
  // a default configuration.
  bool is_default_config_ : 1;

  // True if this instruction has already been detached from its user and
  // operands.
  bool cleaned_up_ : 1;

  // Intrusive flag used by HloComputation, whether this instruction has
  // been marked as dead.
  bool marked_as_dead_ : 1;

  // True if this instruction is the root of a computation.
  bool is_root_ : 1;

  // Instruction operands.
  InstructionVector operands_;

  // If needed, points off to allocated struct holding out-of-line info
  // for things that are rarely filled
  std::unique_ptr<Rare> rare_;

  // The users of this instruction. Users are HLOs where this instruction is an
  // operand.
  Users users_;

  // The computation in which this instruction is contained.
  HloComputation* parent_ = nullptr;

  // The sharding, if one exists.
  // Uses std::shared_ptr to allow reuse of the same sharding object between
  // HloInstructions and other components as HloSharding can be very large for
  // many element tuples.
  std::shared_ptr<const HloSharding> sharding_;

  // Result shape of this instruction.
  Shape shape_;

  // The backend-specific configuration for how a backend should compile this
  // HLO. See the documentation on backend_config().
  BackendConfigWrapper backend_config_;

  // String identifier for instruction.
  std::string name_;

  // Original value this instruction corresponds to in the unoptimized HLO
  // graph.
  std::shared_ptr<OriginalValue> original_value_ = nullptr;

  // Metadata for debugging.  Allocate it on heap, so that it does not increase
  // the memory footprint of HloInstruction.
  std::unique_ptr<OpMetadata> metadata_ = std::make_unique<OpMetadata>();
};

// Explicit instantiations in hlo_instruction.cc.
extern template absl::Status HloInstruction::Accept(DfsHloVisitor*, bool, bool,
                                                    bool);
extern template absl::Status HloInstruction::Accept(ConstDfsHloVisitor*, bool,
                                                    bool, bool);
extern template absl::Status HloInstruction::Visit(DfsHloVisitor* visitor);
extern template absl::Status HloInstruction::Visit(ConstDfsHloVisitor* visitor);

absl::string_view ToString(HloInstruction::FusionKind kind);
absl::StatusOr<HloInstruction::FusionKind> StringToFusionKind(
    absl::string_view kind_name);

// Custom (de)stringification functions for protos that live inside
// HloInstruction.
std::string PaddingConfigToString(const PaddingConfig& padding);

// Returns string representation of frontend attributes.
// Frontend attribute is a list of attribute=<value> pairs where value is either
// a "string" or a JSON-like dict surrounded in {}. Similar to custom_call
// backend config, this can be used to store stringified MLIR-dictionaries with
// pretty printing.
std::string FrontendAttributesToString(
    const FrontendAttributes& frontend_attributes);
std::string StatisticsVizToString(const StatisticsViz& statistics_viz);
std::string ResultAccuracyToleranceToString(
    const ResultAccuracy::Tolerance& tolerance);
std::string RandomAlgorithmToString(const RandomAlgorithm& algorithm);
std::string RandomDistributionToString(const RandomDistribution& distribution);
std::string PrecisionToString(const PrecisionConfig::Precision& precision);
std::string ResultAccuracyToString(ResultAccuracy::Mode accuracy_mode);
std::string AlgorithmToString(const PrecisionConfig::Algorithm& algorithm);
std::string DotDimensionNumbersToString(const DotDimensionNumbers& dnums);
std::string RaggedDotDimensionNumbersToString(
    const RaggedDotDimensionNumbers& dnums);
std::string ConvolutionDimensionNumbersToString(
    const ConvolutionDimensionNumbers& dnums);

absl::StatusOr<RandomAlgorithm> StringToRandomAlgorithm(
    const std::string& name);
absl::StatusOr<RandomDistribution> StringToRandomDistribution(
    const std::string& name);
absl::StatusOr<PrecisionConfig::Precision> StringToPrecision(
    const std::string& name);
absl::StatusOr<PrecisionConfig::Algorithm> StringToAlgorithm(
    const std::string& name);
absl::StatusOr<ResultAccuracy::Mode> StringToResultAccuracy(
    absl::string_view name);
absl::StatusOr<CustomCallSchedule> StringToCustomCallSchedule(
    absl::string_view name);
absl::StatusOr<CustomCallApiVersion> StringToCustomCallApiVersion(
    absl::string_view name);

std::ostream& operator<<(std::ostream& os, HloInstruction::FusionKind kind);

bool IsUnaryOpWithResultAccuracy(HloOpcode opcode);
bool IsValidResultAccuracy(const ResultAccuracy& result_accuracy);
// Map classes that guarantee a deterministic iteration order when the key is
// an HloInstruction* or a const HloInstruction*.
// To make the iteration order over the map deterministic, the comparator
// should not be using the pointer values, but rather an intrinsic property of
// the hlo. Exception: null pointer values compare less than non-null.
struct HloPtrComparator {
  bool operator()(const HloInstruction* const& lhs,
                  const HloInstruction* const& rhs) const;
};

template <typename ValueT>
using HloInstructionMap = std::map<HloInstruction*, ValueT, HloPtrComparator>;

template <typename ValueT>
using ConstHloInstructionMap =
    std::map<const HloInstruction*, ValueT, HloPtrComparator>;

using HloInstructionSet = std::set<HloInstruction*, HloPtrComparator>;
using ConstHloInstructionSet =
    std::set<const HloInstruction*, HloPtrComparator>;

template <HloOpcode op, HloOpcode... rest>
bool HloPredicateIsOp(const HloInstruction* instruction) {
  return (instruction->opcode() == op) ||
         ((instruction->opcode() == rest) || ...);
}

template <HloOpcode op, HloOpcode... rest>
bool HloPredicateIsNotOp(const HloInstruction* instruction) {
  return (instruction->opcode() != op) &&
         ((instruction->opcode() != rest) && ...);
}

/* static */ inline bool HloInstruction::MightHaveCalledComputations(
    HloOpcode opcode) {
  switch (opcode) {
    // Control flow opcodes
    case HloOpcode::kWhile:
    case HloOpcode::kConditional:

    // Fusion contains a sub-computation
    case HloOpcode::kFusion:

    // Async
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:

    // Opcodes for which has_to_apply can return true
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kCall:
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSort:
    case HloOpcode::kTopK:
    case HloOpcode::kCustomCall:
      return true;
    default:
      return false;
  }
}

}  // namespace xla

#endif  // XLA_HLO_IR_HLO_INSTRUCTION_H_
