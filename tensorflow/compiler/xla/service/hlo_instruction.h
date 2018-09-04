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

// HLO instructions are in DAG form and represent the computations that the user
// has built up via the XLA service interface. They are ultimately lowered
// in a platform-aware way by traversing the HLO DAG and emitting a lowered
// form; e.g. see DfsHloVisitor.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTION_H_

#include <functional>
#include <iosfwd>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/iterator_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_clone_context.h"
#include "tensorflow/compiler/xla/service/hlo_domain_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/iterator_range.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class HloComputation;
class HloModule;

// A bunch of switches that control how the hlo text should be printed.
class HloPrintOptions {
 public:
  enum class PrintSubcomputationMode {
    kOff,         // Do not print anything about subcomputations.
    kNameOnly,    // Only print the name of subcomputations.
    kFullBodies,  // Print the full bodies of subcomputations.
  };

  // Constructs the default print options: don't print large constants, don't
  // compact operands, no indentation.
  HloPrintOptions()
      : print_large_constants_(false),
        print_subcomputation_mode_(PrintSubcomputationMode::kNameOnly),
        print_metadata_(true),
        print_backend_config_(true),
        compact_operands_(false),
        print_operand_shape_(true),
        print_program_shape_(true),
        print_percent_(true),
        canonicalize_instruction_names_(false),
        indent_amount_(0),
        is_in_nested_computation_(false) {}

  static HloPrintOptions ShortParsable() {
    return HloPrintOptions()
        .set_print_large_constants(true)
        .set_print_subcomputation_mode(PrintSubcomputationMode::kNameOnly)
        .set_print_metadata(false)
        .set_print_backend_config(false)
        .set_print_operand_shape(false)
        .set_print_program_shape(false)
        .set_print_percent(false);
  }

  // Options to produce the canonical string representing an isomorphic
  // computation graph.
  static HloPrintOptions Canonical() {
    return HloPrintOptions()
        .set_print_subcomputation_mode(PrintSubcomputationMode::kFullBodies)
        .set_print_metadata(false)
        .set_print_backend_config(false)
        .set_compact_operands(true)
        .set_print_operand_shape(true)
        .set_print_program_shape(false)
        .set_print_percent(false)
        .set_canonicalize_instruction_names(true);
  }

  // If true, large constants will be printed out.
  HloPrintOptions& set_print_large_constants(bool value) {
    print_large_constants_ = value;
    return *this;
  }

  HloPrintOptions& set_print_subcomputation_mode(
      PrintSubcomputationMode value) {
    print_subcomputation_mode_ = value;
    return *this;
  }

  // If true, metadata will be printed.
  HloPrintOptions& set_print_metadata(bool value) {
    print_metadata_ = value;
    return *this;
  }

  // If true, backend_config will be printed.
  HloPrintOptions& set_print_backend_config(bool value) {
    print_backend_config_ = value;
    return *this;
  }

  // If true, operands' shapes will be printed.
  HloPrintOptions& set_print_operand_shape(bool value) {
    print_operand_shape_ = value;
    return *this;
  }

  // If true, program shape of hlo computations will be printed.
  HloPrintOptions& set_print_program_shape(bool value) {
    print_program_shape_ = value;
    return *this;
  }

  // If true, names will be printed with prefix '%'.
  HloPrintOptions& set_print_percent(bool value) {
    print_percent_ = value;
    return *this;
  }

  // If true, only a part of operands will be printed out, and their names will
  // be omitted (note that in this case the text will not be parsable).
  HloPrintOptions& set_compact_operands(bool value) {
    compact_operands_ = value;
    return *this;
  }

  // If true, canonicalizes instructions' name. Instead of using "%foo.1" as
  // the name of an instruction, we use "%tmp_1", "%tmp_2" etc.
  HloPrintOptions& set_canonicalize_instruction_names(bool value) {
    canonicalize_instruction_names_ = value;
    return *this;
  }

  // The indent of the hlo text block.
  HloPrintOptions& set_indent_amount(int value) {
    indent_amount_ = value;
    return *this;
  }

  // If true, indicates the instruction being printed is inside a nested
  // computation.
  HloPrintOptions& set_is_in_nested_computation(bool value) {
    is_in_nested_computation_ = value;
    return *this;
  }

  bool print_large_constants() const { return print_large_constants_; }
  PrintSubcomputationMode print_subcomputation_mode() const {
    return print_subcomputation_mode_;
  }
  bool print_metadata() const { return print_metadata_; }
  bool print_backend_config() const { return print_backend_config_; }
  bool compact_operands() const { return compact_operands_; }
  bool print_operand_shape() const { return print_operand_shape_; }
  bool print_program_shape() const { return print_program_shape_; }
  bool print_percent() const { return print_percent_; }
  bool canonicalize_instruction_names() const {
    return canonicalize_instruction_names_;
  }
  int indent_amount() const { return indent_amount_; }
  int is_in_nested_computation() const { return is_in_nested_computation_; }

 private:
  bool print_large_constants_;
  PrintSubcomputationMode print_subcomputation_mode_;
  bool print_metadata_;
  bool print_backend_config_;
  bool compact_operands_;
  bool print_operand_shape_;
  bool print_program_shape_;
  bool print_percent_;
  bool canonicalize_instruction_names_;
  int indent_amount_;
  bool is_in_nested_computation_;
};

// For canonical string output, we need to have a canonical way to rename
// each instruction and its operands. Each operand is renamed as "tmp_<xxx>",
// where <xxx> is an index starting from 0.
class CanonicalNameMap {
 public:
  CanonicalNameMap() : index(0) {}

  string LookupOrInsert(const string& old_name) {
    auto iter = canonical_name_map.find(old_name);
    if (iter != canonical_name_map.end()) {
      return iter->second;
    }

    string new_name = absl::StrCat("tmp_", index++);
    canonical_name_map[old_name] = new_name;
    return new_name;
  }
  void Clear() {
    canonical_name_map.clear();
    index = 0;
  }

 private:
  int64 index;
  tensorflow::gtl::FlatMap<string, string> canonical_name_map;
};

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

  virtual ~HloInstruction();

  // Creates an instruction from the given proto. Arguments:
  //
  //   proto: the proto to convert from.
  //   instruction_map: a map from instruction id to HloInstruction*. This map
  //     must contain all operands of the newly constructed instruction.
  //   computation_map: a map from computation id to HloComputation*. This map
  //     must contain all computations which the newly constructed instruction
  //     calls.
  static StatusOr<std::unique_ptr<HloInstruction>> CreateFromProto(
      const HloInstructionProto& proto,
      const tensorflow::gtl::FlatMap<int64, HloInstruction*>& instruction_map,
      const tensorflow::gtl::FlatMap<int64, HloComputation*>& computation_map);

  // Creates a parameter-retrieving instruction.
  static std::unique_ptr<HloInstruction> CreateParameter(int64 parameter_number,
                                                         const Shape& shape,
                                                         const string& name);

  // Creates a literal constant instruction.
  static std::unique_ptr<HloInstruction> CreateConstant(
      std::unique_ptr<Literal> literal);

  // Creates an Iota instruction.
  static std::unique_ptr<HloInstruction> CreateIota(const Shape& shape,
                                                    int64 iota_dimension);

  // Creates a get tuple element instruction.
  static std::unique_ptr<HloInstruction> CreateGetTupleElement(
      const Shape& shape, HloInstruction* operand, int64 index);

  // Creates a trace instruction that logs the input operand in the computation.
  static std::unique_ptr<HloInstruction> CreateTrace(const string& tag,
                                                     HloInstruction* operand);

  // Creates a random number generation instruction that fills a shape with
  // random numbers from a given distribution.
  static std::unique_ptr<HloInstruction> CreateRng(
      const Shape& shape, RandomDistribution distribution,
      absl::Span<HloInstruction* const> parameters);

  // Creates a unary instruction (one operand).
  // Precondition: opcode must be a legitimate unary operation.
  static std::unique_ptr<HloInstruction> CreateUnary(const Shape& shape,
                                                     HloOpcode opcode,
                                                     HloInstruction* operand);

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
      const Window& window,
      const ConvolutionDimensionNumbers& dimension_numbers,
      int64 feature_group_count = 1);

  // Creates an FFT op, of the type indicated by fft_type.
  static std::unique_ptr<HloInstruction> CreateFft(
      const Shape& shape, HloInstruction* operand, FftType fft_type,
      absl::Span<const int64> fft_length);

  // Creates a dot op with operands 'lhs' and 'rhs' with contracting and batch
  // dimensions specified in 'dimension_numbers'.
  static std::unique_ptr<HloInstruction> CreateDot(
      const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
      const DotDimensionNumbers& dimension_numbers);

  // Creates a dot op with operands 'lhs' and 'rhs' that contracts dimension 1
  // of the LHS with dimension 0 of the RHS with no batch dimensions.  Both LHS
  // and the RHS must be of rank 2.
  static std::unique_ptr<HloInstruction> CreateCanonicalDot(
      const Shape& shape, HloInstruction* lhs, HloInstruction* rhs);

  // Creates a reduce-precision op, where operand is the data to reduce in
  // precision, and exponent_bits and mantissa_bits describe the precision to
  // reduce it to.
  static std::unique_ptr<HloInstruction> CreateReducePrecision(
      const Shape& shape, HloInstruction* operand, const int exponent_bits,
      const int mantissa_bits);

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
  // `all_reduce_id`: for Allreduce nodes from different modules, if they have
  // the same all_reduce_id, they will be 'Allreduce'd. If empty, Allreduce will
  // not be applied cross modules.
  //
  // TODO(b/79737069): Rename this to AllReduce.
  static std::unique_ptr<HloInstruction> CreateCrossReplicaSum(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      const std::vector<ReplicaGroup>& replica_groups,
      absl::string_view barrier, const absl::optional<int64>& all_reduce_id);

  // This op handles the communication of an Alltoall operation. On each core,
  // the operands are N ops in the same shape, where N is the number of cores
  // participating the Alltoall. Then the N operands are scattered to N cores,
  // e.g., the ith operand is sent to the ith core. Then each core gathers the
  // received data into a tuple.
  //
  // - `replica_groups`: each ReplicaGroup contains a list of replica id. If
  // empty, all replicas belong to one group in the order of 0 - (n-1). Alltoall
  // will be applied within subgroups in the specified order. For example,
  // replica groups = {{1,2,3},{4,5,0}} means, an Alltoall will be applied
  // within replica 1, 2, 3, and in the gather phase, the received blocks will
  // be concatenated in the order of 1, 2, 3; another Alltoall will be applied
  // within replica 4, 5, 0, and the concatenation order is 4, 5, 0.
  static std::unique_ptr<HloInstruction> CreateAllToAll(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      const std::vector<ReplicaGroup>& replica_groups);

  // Creates a communitation instructions that permutes data cross replicas.
  // Data is sent/received according to the (source_replica_id,
  // target_replica_id) pairs in `source_target_pairs`. If a replica id is not a
  // target_replica_id in any pair, the output on that replica is a tensor
  // conssits of 0(s) in `shape`.
  static std::unique_ptr<HloInstruction> CreateCollectivePermute(
      const Shape& shape, HloInstruction* operand,
      const std::vector<std::pair<int64, int64>>& source_target_pairs);

  // Creates a conversion instruction, where operand is the data to convert and
  // shape is the target shape for the conversion.
  static std::unique_ptr<HloInstruction> CreateConvert(const Shape& shape,
                                                       HloInstruction* operand);

  // Creates a bitcast conversion instruction, where operand is the data to
  // convert and shape is the target shape for the conversion.
  static std::unique_ptr<HloInstruction> CreateBitcastConvert(
      const Shape& shape, HloInstruction* operand);

  // Creates an infeed instruction, which reads data of the given shape from the
  // Infeed interface of the device. infeed_shape is the shape of the data
  // received from the infeed *not* the shape of the infeed instruction which
  // is a tuple containing the infeed_shape and the TOKEN.
  static std::unique_ptr<HloInstruction> CreateInfeed(
      const Shape& infeed_shape, HloInstruction* token_operand,
      const string& config);

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
      HloInstruction* operand, HloInstruction* token, int64 channel_id,
      bool is_host_transfer = false);

  // Blocks until data transfer for the Send instruction (operand) is complete.
  // The operand must be kSend.
  static std::unique_ptr<HloInstruction> CreateSendDone(
      HloInstruction* operand, bool is_host_transfer = false);

  // Creates an asynchronous receive instruction with the given channel id,
  // which allocates resources to receive data of the given shape from a unique
  // send instruction in another computation that has the same channel id.  If
  // is_host_transfer is true, then this Send operation transfers data from the
  // host.
  static std::unique_ptr<HloInstruction> CreateRecv(
      const Shape& shape, HloInstruction* token, int64 channel_id,
      bool is_host_transfer = false);

  // Blocks until data transfer for the Recv instruction (operand) is complete
  // and returns the receive buffer. The operand must be kRecv.
  static std::unique_ptr<HloInstruction> CreateRecvDone(
      HloInstruction* operand, bool is_host_transfer = false);

  // Creates a slice instruction, where the operand is sliced by the given
  // start/limit indices.
  static std::unique_ptr<HloInstruction> CreateSlice(
      const Shape& shape, HloInstruction* operand,
      absl::Span<const int64> start_indices,
      absl::Span<const int64> limit_indices, absl::Span<const int64> strides);

  // Creates a slice instruction, where the first operand is sliced by
  // start indices specified in the second operand, and by size specified in
  // 'slice_sizes'.
  static std::unique_ptr<HloInstruction> CreateDynamicSlice(
      const Shape& shape, HloInstruction* operand,
      HloInstruction* start_indices, absl::Span<const int64> slice_sizes);

  // Creates a dynamic update slice instruction, which updates a slice
  // of 'operand' with 'update' and 'start_indices'.
  static std::unique_ptr<HloInstruction> CreateDynamicUpdateSlice(
      const Shape& shape, HloInstruction* operand, HloInstruction* update,
      HloInstruction* start_indices);

  // Creates a concatenate instruction, where the operands are concatenated on
  // the provided dimension.
  static std::unique_ptr<HloInstruction> CreateConcatenate(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      int64 dimension);

  // Creates a reduce instruction, where the computation (given by the handle)
  // is applied successively to every element in operand. For example, let f be
  // the function to apply, which takes 2 arguments, an accumulator and the
  // current value. Let init be an initial value (which is normally chosen to be
  // the identity element for f, e.g. 0 if f is addition).
  // Then the reduce HLO will compute:
  // f(f(init, value0), value1), ...)
  static std::unique_ptr<HloInstruction> CreateReduce(
      const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
      absl::Span<const int64> dimensions_to_reduce,
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
  // TODO(b/112040122): Add support to this in HLO passes and in backends.
  static std::unique_ptr<HloInstruction> CreateReduce(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::Span<HloInstruction* const> init_values,
      absl::Span<const int64> dimensions_to_reduce,
      HloComputation* reduce_computation);

  // Creates a reduce-window instruction, where the computation (given
  // by the handle) is applied window-wise at each valid window
  // position in the operand.
  static std::unique_ptr<HloInstruction> CreateReduceWindow(
      const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
      const Window& window, HloComputation* reduce_computation);

  // Creates a batch-norm-training instruction.
  static std::unique_ptr<HloInstruction> CreateBatchNormTraining(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* offset, float epsilon, int64 feature_index);

  // Creates a batch-norm-inference instruction.
  static std::unique_ptr<HloInstruction> CreateBatchNormInference(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* offset, HloInstruction* mean, HloInstruction* variance,
      float epsilon, int64 feature_index);

  // Creates a batch-norm-grad instruction.
  static std::unique_ptr<HloInstruction> CreateBatchNormGrad(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* mean, HloInstruction* variance,
      HloInstruction* grad_output, float epsilon, int64 feature_index);

  // Creates a scatter computation that scatters the `source` array to the
  // selected indices of each window.
  static std::unique_ptr<HloInstruction> CreateSelectAndScatter(
      const Shape& shape, HloInstruction* operand, HloComputation* select,
      const Window& window, HloInstruction* source, HloInstruction* init_value,
      HloComputation* scatter);

  // Creates a broadcast instruction.
  static std::unique_ptr<HloInstruction> CreateBroadcast(
      const Shape& shape, HloInstruction* operand,
      absl::Span<const int64> broadcast_dimensions);

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
      const std::function<HloInstruction*(std::unique_ptr<HloInstruction>)>&
          adder);

  // Creates a pad instruction, where the operand is padded on the edges and
  // between the elements with the given padding value.
  static std::unique_ptr<HloInstruction> CreatePad(
      const Shape& shape, HloInstruction* operand,
      HloInstruction* padding_value, const PaddingConfig& padding_config);

  // Creates a reshape instruction, where the operand is flattened row-major
  // order and then reshaped to the given result shape.
  static std::unique_ptr<HloInstruction> CreateReshape(const Shape& shape,
                                                       HloInstruction* operand);

  // Creates a transpose instruction which permutes the operand dimensions.
  static std::unique_ptr<HloInstruction> CreateTranspose(
      const Shape& shape, HloInstruction* operand,
      absl::Span<const int64> dimensions);

  // Creates a sort op, with a keys operand, and an optional values operand.
  static std::unique_ptr<HloInstruction> CreateSort(
      const Shape& shape, int64 dimension, HloInstruction* keys,
      HloInstruction* values = nullptr);

  // Creates a while instruction, given a condition computation, a body
  // computation, and the initial value for the input of the computations. For
  // example, shape: S32, condition: i -> i < 1000, body: i -> i * 2, init: 1
  // corresponds to the C code below.
  // int32 i = 1; int32 result = while(i < 1000) { i = i * 2 }
  static std::unique_ptr<HloInstruction> CreateWhile(const Shape& shape,
                                                     HloComputation* condition,
                                                     HloComputation* body,
                                                     HloInstruction* init);

  static std::unique_ptr<HloInstruction> CreateConditional(
      const Shape& shape, HloInstruction* pred,
      HloInstruction* true_computation_arg, HloComputation* true_computation,
      HloInstruction* false_computation_arg, HloComputation* false_computation);

  static std::unique_ptr<HloInstruction> CreateGather(
      const Shape& shape, HloInstruction* operand,
      HloInstruction* start_indices,
      const GatherDimensionNumbers& gather_dim_numbers,
      absl::Span<const int64> slice_sizes);

  static std::unique_ptr<HloInstruction> CreateScatter(
      const Shape& shape, HloInstruction* operand,
      HloInstruction* scatter_indices, HloInstruction* updates,
      HloComputation* update_computation,
      const ScatterDimensionNumbers& scatter_dim_numbers);

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
      const Shape& shape, FusionKind fusion_kind, HloInstruction* fused_root);

  static std::unique_ptr<HloInstruction> CreateFusion(
      const Shape& shape, FusionKind fusion_kind,
      absl::Span<HloInstruction* const> operands,
      HloComputation* fusion_computation);

  // Creates a call instruction that applies the given computation on the given
  // operands. "shape" is the resultant shape.
  static std::unique_ptr<HloInstruction> CreateCall(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* computation);

  // Creates a custom call instruction that applies the given custom call target
  // to the given operands. "shape" is the resultant shape.
  static std::unique_ptr<HloInstruction> CreateCustomCall(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::string_view custom_call_target);

  // Creates a tuple instruction with the given elements. This is a convenience
  // wrapper around CreateVariadic.
  static std::unique_ptr<HloInstruction> CreateTuple(
      absl::Span<HloInstruction* const> elements);

  // Creates a reverse instruction, which reverses the order of the elements
  // in the specified dimensions.
  static std::unique_ptr<HloInstruction> CreateReverse(
      const Shape& shape, HloInstruction* operand,
      absl::Span<const int64> dimensions);

  // Creates a Afterall instruction used for joining or creating new values of
  // token type which thread through side-effecting operations. Operands must
  // all be tokens, and there must be at least one operand.
  static std::unique_ptr<HloInstruction> CreateAfterAll(
      absl::Span<HloInstruction* const> operands);

  // Creates an AfterAll instruction which creates a token type out of thin air
  // (no operands). This is a separate method from CreateAfterAll to facility
  // the removal of operand-less AfterAll instructions.
  // TODO(b/110532604): Remove this capability of creating a token from nothing
  // when we plumb a primordial token from the entry computation.
  static std::unique_ptr<HloInstruction> CreateToken();

  // Returns the opcode for this instruction.
  HloOpcode opcode() const { return opcode_; }

  // Returns true if this instruction has a side effect, irrespective of whether
  // any called computations may contain an instruction with side effects.
  bool HasSideEffectNoRecurse() const;

  // Returns true if this instruction has a side effect. An instruction has a
  // side effect if it uses certain opcodes or calls a computation with a side
  // effect.
  bool HasSideEffect() const;

  // Returns the result shape of this instruction.
  const Shape& shape() const;

  // Returns the (mutable) result shape of this instruction.
  Shape* mutable_shape() { return &shape_; }

  // Returns the ith operand to this instruction.
  const HloInstruction* operand(int64 i) const;

  // Returns the ith operand to this instruction.
  HloInstruction* mutable_operand(int64 i);

  // Returns the number of operands to this instruction.
  int64 operand_count() const { return operands_.size(); }

  // Returns the vector of operands of this instruction.
  using InstructionVector = absl::InlinedVector<HloInstruction*, 2>;
  const InstructionVector& operands() const { return operands_; }

  // Returns the vector of unique operands, in the same order they are found
  // within the operand vector.
  InstructionVector unique_operands() const;

  // Returns the index of 'target' in the operands sequence.
  // Precondition: target must be an operand (or a fatal error will occur).
  int64 operand_index(const HloInstruction* target) const;

  // Returns the number of users of this instruction.
  int64 user_count() const { return users_.size(); }

  // Returns the users of this instruction.
  const std::vector<HloInstruction*>& users() const { return users_; }

  // Returns true if this instruction is a user of 'instruction'.
  bool IsUserOf(const HloInstruction* instruction) const {
    return ContainsKey(instruction->user_set_, this);
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
  Status AddControlDependencyTo(HloInstruction* instruction);

  // Removes a previously added control dependency from this instruction to
  // 'instruction'.
  Status RemoveControlDependencyTo(HloInstruction* instruction);

  // Drops all control predecessors and successors from this HLO instruction.
  Status DropAllControlDeps();

  // Copies the control predecessors and successors on this HLO instruction to
  // `inst`.  Does not do a deep copy so this makes sense only if `inst` and
  // this HLO are in the same module.
  //
  // Depending on the use cases we see in practice, in the future we may
  // consider folding the logic here into Clone, CloneWithNewOperands and
  // ReplaceAllUsesWith by treating control dependencies like data dependencies.
  Status CopyAllControlDepsFrom(const HloInstruction* inst);

  // Returns the set of control predecessors (successors) of this
  // instruction. Control predecessors (successors) must execute before (after)
  // the current instruction.
  const std::vector<HloInstruction*>& control_predecessors() const {
    return control_predecessors_;
  }
  const std::vector<HloInstruction*>& control_successors() const {
    return control_successors_;
  }

  // Returns true if "other" performs the same computation as this instruction.
  bool Identical(
      const HloInstruction& other,
      const std::function<bool(const HloInstruction*, const HloInstruction*)>&
          eq_operands = std::equal_to<const HloInstruction*>(),
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations = std::equal_to<const HloComputation*>(),
      bool layout_sensitive = true) const {
    // An instruction is always identical to itself.
    if (this == &other) {
      return true;
    }

    // Identical instruction must have the same opcode, shape, and identical
    // operands.
    if (opcode() != other.opcode()) {
      return false;
    }
    if (!(layout_sensitive ? ShapeUtil::Equal(shape(), other.shape())
                           : ShapeUtil::Compatible(shape(), other.shape()))) {
      return false;
    }
    if (operands().size() != other.operands().size()) {
      return false;
    }

    // Use an explicit loop rather than ContainerEquals, because copying around
    // std::functions may be too expensive in some cases.
    for (size_t i = 0; i < operands().size(); ++i) {
      if (!eq_operands(operand(i), other.operand(i))) {
        return false;
      }
    }

    if (backend_config_ != other.backend_config_) {
      return false;
    }

    if (!absl::c_equal(precision_config_.operand_precision(),
                       other.precision_config_.operand_precision())) {
      return false;
    }

    return IdenticalSlowPath(other, eq_computations);
  }

  // Returns whether the instruction has a constant operand.
  bool HasConstantOperand() const;

  // Replaces the use of this instruction in "user" with "new_producer". Note
  // that there might be multiple uses of this instruction in "user"; all will
  // be replaced.
  //
  // If user is a fusion instruction, this function will remove any duplicated
  // operands of it which could be created due to this replacement.
  Status ReplaceUseWith(HloInstruction* user, HloInstruction* new_producer);

  // Replaces the specified operand with new_operand.
  //
  // This function does NOT remove duplicated operands even if this instruction
  // is a fusion, so that the existing operand numbers do not change.
  Status ReplaceOperandWith(int64 operand_no, HloInstruction* new_operand);

  // Replaces all uses of this instruction with the new producer. If
  // new_producer is a user of this instruction then new_producer remains a use
  // of this instruction to avoid introducing cycles into the graph.
  //
  // If this instruction is the root of its computation, sets the computation's
  // root to new_producer.
  //
  // If a user is a fusion instruction, this function will remove any duplicated
  // operands of it which could be created due to this replacement.
  Status ReplaceAllUsesWith(HloInstruction* new_producer);

  // Performs a postorder DFS visit using this node as the root. If
  // call_finish_visit is true, then DfsHloVisitor::FinishVisit is called when
  // complete. If ignore_control_predecessors is true, instructions only
  // reachable via control dependencies will not be visited, and the postorder
  // will not take control dependencies into account. It is as if the control
  // dependencies didn't exist in the graph at all.
  template <typename HloInstructionPtr>
  Status Accept(DfsHloVisitorBase<HloInstructionPtr>* visitor,
                bool call_finish_visit = true,
                bool ignore_control_predecessors = false);
  Status Accept(ConstDfsHloVisitor* visitor, bool call_finish_visit = true,
                bool ignore_control_predecessors = false) const {
    return const_cast<HloInstruction*>(this)->Accept(
        visitor, call_finish_visit, ignore_control_predecessors);
  }

  // Same as Accept() above, but the order of operand and control predecessor
  // visitation is determined by the given operand order; if compare(A, B) ==
  // true, A is visited before B.
  using CompareFunction =
      std::function<bool(const HloInstruction*, const HloInstruction*)>;
  Status AcceptWithOperandOrder(DfsHloVisitor* visitor,
                                const CompareFunction& operand_order,
                                bool call_finish_visit = true);

  // Performs a postorder DFS visit using this node as the root. Calls the given
  // visitor function at each instruction.
  Status Accept(const std::function<Status(HloInstruction*)>& visitor_func);
  Status Accept(
      const std::function<Status(const HloInstruction*)>& visitor_func) const;

  // Visits all instructions rooted at this instruction using the given visitor
  // in the given order. 'order' must contain at least the set of instructions
  // rooted at this node (ie, those accessible from a DFS traversal from this
  // instruction). Instructions contained in 'order' which are not in the set of
  // instructions rooted at this node are ignored. 'order' must also be a valid
  // topological sort of these instructions (defs appear before uses) though
  // need not be a DFS post-order.
  Status AcceptOrdered(DfsHloVisitor* visitor,
                       const std::vector<const HloInstruction*>& order);

  // Visit this instruction and only this instruction with the given visitor.
  template <typename HloInstructionPtr>
  Status Visit(DfsHloVisitorBase<HloInstructionPtr>* visitor);

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

  // Gets/sets the to_apply HloComputation for Call, Map, Reduce, etc.
  // The setter should only be called by HloModule or HloComputation methods.
  //
  // Precondition: The instruction has a valid to_apply_ field.
  HloComputation* to_apply() const;
  void set_to_apply(HloComputation* to_apply);

  // Gets/sets the while_condition or while_body HloComputation for While. The
  // setters should only be called by HloModule or HloComputation methods.
  //
  // Precondition: The instruction is a While instruction.
  HloComputation* while_condition() const;
  HloComputation* while_body() const;
  void set_while_condition(HloComputation* while_condition);
  void set_while_body(HloComputation* while_body);

  // Gets/sets the true and false HloComputation for Conditional. The setters
  // should only be called by HloModule or HloComputation methods.
  //
  // Precondition: The instruction is a Conditional instruction.
  HloComputation* true_computation() const;
  HloComputation* false_computation() const;
  void set_true_computation(HloComputation* true_computation);
  void set_false_computation(HloComputation* false_computation);

  // Returns a string for the signature of this instruction if considered as a
  // function, e.g. the signature of an F32 add is (F32, F32) -> F32.
  string SignatureString() const;

  // Returns a debugging string that represents this instruction.
  //
  // (We express the default options using an overload rather than a default
  // param because gdb ignores default params, but does resolve overloads.)
  //
  // TODO(b/73348663): Make ToString() adaptive to the size of the string by
  // default, backing off on providing full information for very large strings,
  // or provide a different name for a ToString-like function that does that.
  string ToString() const { return ToString(HloPrintOptions()); }
  string ToString(const HloPrintOptions& options) const;

  // Components of the ToString() representation:

  // Returns a string representation of the operand list.
  string OperandsToString(const HloPrintOptions& options) const;

  // Returns string representation of op-specific attributes.
  std::vector<string> ExtraAttributesToString(
      const HloPrintOptions& options) const;

  // As ToString, but returns a shorter string.
  string ToShortString() const;

  // Returns a serialized representation of this instruction.
  virtual HloInstructionProto ToProto() const;

  // Returns a category for the HLO. This could be something like "convolution"
  // or "elementwise".
  virtual string ToCategory() const;

  // Returns a logging instruction, if the output of this instruction is logged.
  //
  // Postcondition: retval == nullptr || retval->opcode() == HloOpcode::kTrace
  HloInstruction* tracing() const;
  void set_tracing(HloInstruction* trace_instruction);

  // Returns true if this instruction is fused, ie contained within a fusion
  // instruction.
  bool IsFused() const;

  // Returns true if this instruction can be legally fused into a fusion
  // instruction.
  bool IsFusible() const;

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
  absl::optional<int64> sharding_unique_device() const {
    if (sharding_ == nullptr) {
      return absl::optional<int64>();
    }
    return sharding_->UniqueDevice();
  }
  // Sets the sharding of this operator. Should only be called by HloModule or
  // HloComputation methods.
  void set_sharding(const HloSharding& sharding) {
    sharding_ = std::make_shared<const HloSharding>(sharding);
  }
  void set_sharding(std::shared_ptr<const HloSharding> sharding) {
    sharding_ = std::move(sharding);
  }
  void set_single_sharding(const HloSharding& sharding);
  // Sets a sharding that assigns the current instruction to device.
  void set_device_sharding(int64 device) {
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

  // Retrieves the operand side metadata of a kDomain instruction.
  const DomainMetadata& operand_side_metadata() const {
    return *operand_side_metadata_;
  }
  // Retrieves the user side metadata of a kDomain instruction.
  const DomainMetadata& user_side_metadata() const {
    return *user_side_metadata_;
  }

  // When creating a new instruction which either replaces, or shifts up (kCopy
  // insertion case), another instruction, we need to make sure the certain
  // properties of the new instruction are copied into the derived one. As of
  // today, the metadata and sharding will be propagated to the derived
  // instruction.
  void SetupDerivedInstruction(HloInstruction* derived_instruction) const;

  // Returns data on the dimension numbers used for a dot operation.
  const DotDimensionNumbers& dot_dimension_numbers() const {
    CHECK(dot_dimension_numbers_ != nullptr);
    return *dot_dimension_numbers_;
  }

  // Returns the dump string of the dot dimension numbers.
  string DotDimensionNumbersToString() const;

  // Returns the dump string of the precision configuration.
  string PrecisionConfigToString() const;

  // Clones the HLO instruction. The clone will have the same opcode, shape, and
  // operands. After creation the clone has no uses. "this" (the instruction
  // cloned from) is not changed. Suffix is the string to append to the name of
  // the instruction to form the name of the cloned instruction.
  // Ignores the control predecessors and successors of this HLO instruction.
  std::unique_ptr<HloInstruction> Clone(
      const string& suffix = "clone", HloCloneContext* context = nullptr) const;

  // Clones the HLO instruction as above but with new shape and operands.
  std::unique_ptr<HloInstruction> CloneWithNewOperands(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context = nullptr) const;

  // Returns the computations this instruction directly calls (if any).
  const std::vector<HloComputation*>& called_computations() const {
    return called_computations_;
  }

  // Replaces all called computations based on a map function. This is needed
  // when we clone hlo_computations and want to let the instructions to point
  // to the newly cloned nodes.
  void ReplaceCalledComputations(
      std::function<HloComputation*(HloComputation*)> map_function) {
    for (int64 i = 0; i < called_computations_.size(); ++i) {
      called_computations_[i] = map_function(called_computations_[i]);
    }
  }

  // Clears out the called computations.
  //
  // This is, in particular, necessary when inlining function bodies into their
  // caller. If there were side-effecting operations in the called computations,
  // the call itself is considered side-effecting and thus cannot be removed. By
  // clearing out the computations, we reflect the fact that all side-effecting
  // properties have been reflected in the caller, and make the call HLO
  // removable.
  void ClearCalledComputations() { called_computations_.clear(); }

  // Returns true if this instruction performs an elementwise operation on
  // `operand_idx`-th operand. An instruction is elementwise on an operand iff,
  // after performing necessary implicit broadcast
  // (cs/IrArray::EmitArrayElementAddress), to compute the output at index
  // {i_0,i_1,...,i_n}, the only element required from the operand (if any) is
  // the element at {i_0,i_1,...,i_n}.
  //
  // Note on performance: when this instruction is kFusion, this method, in the
  // worst case, scans all fused instructions. We could speed this up by
  // caching.
  bool IsElementwiseOnOperand(int64 operand_idx) const;

  // Returns true if this instruction is elementwise on all its operands.
  bool IsElementwise() const;

  // Returns true if this is an cross module all-reduce instrucion.
  bool IsCrossModuleAllReduce() const;

  // Returns true if this elementwise instruction implicitly broadcasts operand
  // `operand_idx`.
  //
  // Precondition: this instruction should be an elementwise operation.
  bool ImplicitlyBroadcastsOperand(int64 operand_idx) const;

  // Returns true if this instruction is binary and elementwise.
  bool IsElementwiseBinary() const;

  // Returns whether this instruction may reuse elements of its `i`th operand.
  bool ReusesOperandElements(int64 i) const {
    return OperandElementUse(i) == UseKind::kReuse;
  }

  // Returns the indices that the given operand appear in the operand list of
  // this instruction. Note that an instruction can use the same operand
  // multiple times.
  std::vector<int64> OperandIndices(const HloInstruction* operand) const;

  // Convenience helper for ShapeUtil::InsertedOrDeleted1SizedDimensions. If
  // this reshape merely inserts or deletes 1-sized dimensions, return the input
  // indices of the deleted dimensions and the output indices of the inserted
  // dimensions.
  //
  // Precondition: this op must be a reshape.
  std::tuple<bool, std::vector<int64>, std::vector<int64>>
  ReshapeMerelyInsertsOrDeletes1SizedDimensions() const;

  // Gets the string identifier for this instruction.
  const string& name() const { return name_; }

  // Sets the string identifier for this instruction. Name will be sanitized to
  // match the regexp "[a-zA-Z_][a-zA-Z0-9_.-]*".
  void SetAndSanitizeName(const string& name) {
    name_ = NameUniquer::GetSanitizedName(name);
  }

  // Use the given NameUniquer to select a unique name for the instruction based
  // on the instruction's existing name.
  void UniquifyName(NameUniquer* name_uniquer);

  // Set the unique id for this instruction to "id"
  void SetUniqueId(int id) {
    CHECK_EQ(unique_id_, -1);  // Should not be assigned already
    CHECK_GE(id, 0);
    unique_id_ = id;
  }

  // Return the unique ID assigned to this node via SetUniqueId (or -1
  // if no id has been assigned yet).
  int unique_id() const { return unique_id_; }

  // Returns the backend-specific configuration for how a backend should compile
  // this HLO. The meaning of the field is backend specific. Not for use before
  // or during general HLO optimization, since HLO optimizations do not preserve
  // this field and they cannot interpret it due to its meaning being backend
  // specific.
  //
  // ConfigProto should be a protobuf Message type.
  template <typename ConfigProto>
  StatusOr<ConfigProto> backend_config() const {
    ConfigProto proto;
    TF_RETURN_IF_ERROR(GetBackendConfigInternal(&proto));
    return std::move(proto);
  }
  Status set_backend_config(const tensorflow::protobuf::Message& proto);

  // Getter/setter for raw JSON-encoded backend config.  Prefer the
  // functions above that deal in proto Messages where possible.
  const string& raw_backend_config_string() const { return backend_config_; }
  void set_raw_backend_config_string(string config_str) {
    backend_config_ = std::move(config_str);
  }

  // Returns a string representation of a proto in the format used by
  // raw_backend_config_string.
  //
  // This is morally equivalent to:
  //
  //   HloInstruction instr;
  //   TF_RETURN_IF_ERROR(instr.set_backend_config(proto));
  //   return instr.raw_backend_config_string();
  //
  static StatusOr<string> BackendConfigToRawString(
      const tensorflow::protobuf::Message& proto);

  // Returns the information used to tell the implementation information about
  // what sort of precision is requested. The meaning of the field is backend
  // specific. At the moment, it is only supported for kConvolution and kDot.
  // Transformations on one kDot or kConvolution to another will preserve this
  // information. Transformations to other HLOs will not preserve this
  // information but it is presumed that the alternate lowering is strictly
  // superior.
  const PrecisionConfigProto& precision_config() const {
    return precision_config_;
  }
  void set_precision_config(const PrecisionConfigProto& precision_config) {
    precision_config_ = precision_config;
  }

  // Sets the debug metadata for this instruction.
  void set_metadata(const OpMetadata& metadata) { metadata_ = metadata; }
  const OpMetadata& metadata() const { return metadata_; }

  // Set/get the computation containing this instruction. set_parent should only
  // be called by HloComputation methods which add/remove instructions to
  // computations.
  void set_parent(HloComputation* computation) { parent_ = computation; }
  const HloComputation* parent() const { return parent_; }
  HloComputation* parent() { return parent_; }

  // Returns the module for this instruction.
  HloModule* GetModule() const;

  // Returns whether we could assign input and output layouts to this
  // instruction to make it a bitcast.
  bool CouldBeBitcast() const;

  // Get/Set the number of partitions per outer dimension (in order, starting
  // with outer-most dimension first). Currently used by the parallel cpu
  // backend to partition HLOs into parallel tasks.
  //
  // TODO(b/62783254) Replace these methods with a more general way to
  // annotate HLOs with backend-specific information.
  const std::vector<int64>& outer_dimension_partitions() const {
    return outer_dimension_partitions_;
  }
  void set_outer_dimension_partitions(
      const std::vector<int64>& outer_dimension_partitions);

  // Old methods kept for smooth subclassing transition BEGIN.
  // TODO(b/80131774): Remove this code.

  // Delegates to HloBatchNormInstruction::feature_index.
  int64 feature_index() const;

  // Delegates to HloBatchNormInstruction::epsilon.
  float epsilon() const;

  // Delegates to HloFftInstruction::fft_type.
  FftType fft_type() const;

  // Delegates to HloFftInstruction::fft_length.
  const std::vector<int64>& fft_length() const;

  // Delegates to HloSendRecvInstruction::channel_id.
  int64 channel_id() const;

  // Returns the dimension sizes or numbers associated with this instruction.
  virtual const std::vector<int64>& dimensions() const {
    LOG(FATAL) << "Unimplemented method.";
  }
  virtual int64 dimensions(int64 index) const {
    LOG(FATAL) << "Unimplemented method.";
  }

  // Delegates to HloConcatenateInstruction::concatenate_dimension.
  int64 concatenate_dimension() const;

  // Returns whether this instruction does a rank-2 transposition.
  bool IsRank2Transpose() const;

  // Delegates to HloSliceInstruction::slice_start.
  int64 slice_starts(int64 dimension) const;
  const std::vector<int64>& slice_starts() const;

  // Delegates to HloSliceInstruction::slice_limits.
  int64 slice_limits(int64 dimension) const;
  const std::vector<int64>& slice_limits() const;

  // Delegates to HloSliceInstruction::slice_strides.
  int64 slice_strides(int64 dimension) const;
  const std::vector<int64>& slice_strides() const;

  // Delegates to HloSliceInstruction::IsInPlaceSlice.
  bool IsInPlaceSlice() const;

  // Returns the literal associated with this instruction.
  const Literal& literal() const;

  // Returns whether the instruction is a constant.
  bool IsConstant() const;

  // Delegate to HloConstantInstruction::RelayoutConstant.
  void RelayoutConstant(const Layout& new_layout,
                        const ShapeIndex& shape_index = {});

  // Delegates to HloTraceInstruction::TracingTag.
  string TracingTag() const;

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
  const tensorflow::gtl::iterator_range<UnwrappingIterator<
      std::list<std::unique_ptr<HloInstruction>>::const_iterator>>
  fused_instructions() const;

  const tensorflow::gtl::iterator_range<
      UnwrappingIterator<std::list<std::unique_ptr<HloInstruction>>::iterator>>
  fused_instructions();

  // Delegates to HloFusionInstruction::fused_instruction_count.
  int64 fused_instruction_count() const;

  // Delegates to HloFusionInstruction::fused_parameter.
  HloInstruction* fused_parameter(int64 parameter_number) const;

  // Delegates to HloFusionInstruction::fused_parameters.
  const std::vector<HloInstruction*>& fused_parameters() const;

  // Returns true if this instruction is a fusion instruction that generates
  // multiple outputs.
  const bool IsMultiOutputFusion() const;

  // Delegates to HloFusionInstruction::fusion_kind.
  FusionKind fusion_kind() const;

  // Delegates to HloFusionInstruction::set_fusion_kind.
  void set_fusion_kind(FusionKind kind);

  // Delegates to HloRngInstruction::random_distribution.
  RandomDistribution random_distribution() const;

  // Delegates to HloParameterInstruction::parameter_number.
  int64 parameter_number() const;

  // Delegates to HloGetTupleElementInstruction::tuple_index.
  int64 tuple_index() const;

  // Delegates to HloReducePrecisionInstruction::exponent_bits.
  int32 exponent_bits() const;

  // Delegates to HloReducePrecisionInstruction::mantissa_bits.
  int32 mantissa_bits() const;

  // Delegates to HloInfeedInstruction::infeed_config.
  string infeed_config() const;

  // Delegates to HloInfeedInstruction::set_infeed_config.
  void set_infeed_config(const string& config);

  // Returns the config for the Outfeed instruction.
  const string& outfeed_config() const;

  // Returns the shape for the Outfeed instruction.
  const Shape& outfeed_shape() const;

  // Delegates to HloCollectiveInstruction::replica_groups.
  const std::vector<ReplicaGroup>& replica_groups() const;

  // Delegates to HloCollectivePermuteInstruction::source_target_pairs.
  const std::vector<std::pair<int64, int64>>& source_target_pairs() const;

  // Delegates to HloAllReduceInstruction::cross_replica_sum_barrier.
  string cross_replica_sum_barrier() const;
  void set_cross_replica_sum_barrier(const string& barrier);

  // Delegates to HloAllReduceInstruction::all_reduce_id.
  absl::optional<int64> all_reduce_id() const;

  // Returns data on the window in a windowed operation such as
  // convolution.
  virtual const Window& window() const {
    LOG(FATAL) << "Unimplemented method.";
  }

  // Sets the window data in a windowed operation such as convolution.
  virtual void set_window(const Window& window) {
    LOG(FATAL) << "Unimplemented method.";
  }

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
  int64 feature_group_count() const;

  void set_feature_group_count(int64 feature_group_count);

  // Delegates to HloSelectAndScatterInstruction::select.
  HloComputation* select() const;

  // Delegates to HloSelectAndScatterInstruction::scatter.
  HloComputation* scatter() const;

  // Delegates to HloSelectAndScatterInstruction::set_select.
  void set_select(HloComputation* computation);

  // Delegates to HloSelectAndScatterInstruction::set_scatter.
  void set_scatter(HloComputation* computation);

  // Delegates to HloCustomCallInstruction::custom_call_target.
  const string& custom_call_target() const;

  // Delegates to HloPadInstruction::padding_config.
  const PaddingConfig& padding_config() const;

  // Delegates to HloDynamicSliceInstruction::slice_sizes.
  int64 slice_sizes(int64 dimension) const;

  // Delegates to HloDynamicSliceInstruction::dynamic_slice_sizes.
  const std::vector<int64>& dynamic_slice_sizes() const;

  // Delegates to HloGatherInstruction::gather_dimension_numbers.
  const GatherDimensionNumbers& gather_dimension_numbers() const;
  // Delegates to HloGatherInstruction::gather_slice_sizes.
  absl::Span<const int64> gather_slice_sizes() const;

  // Delegates to HloScatterInstruction::scatter_dimension_numbers().
  const ScatterDimensionNumbers& scatter_dimension_numbers() const;

  // Old methods kept for smooth subclassing transition END.

 protected:
  enum class UseKind { kNoUse, kReuse, kUsePermutingElements, kUse };
  // Helper class for computing OperandElementUse for kFusion.
  class FusionReusesParamElements;

  // Internal constructor for a given opcode/shape, other fields must be filled
  // by factory methods.
  HloInstruction(HloOpcode opcode, const Shape& shape);

  // Appends operand to the list of operands and adds this instruction as a user
  // of the operand.
  void AppendOperand(HloInstruction* operand);

  void RemoveOperandAt(int index) {
    operands_.erase(operands_.begin() + index);
  }

  // Removes a list of operands with the given indices in ascending order.
  void RemoveOperandsAtAscendingIndices(
      absl::Span<const int> ascending_indices);

  void AppendComputation(HloComputation* computation) {
    called_computations_.push_back(computation);
  }

  void DetachFrom(HloInstruction* usee) { usee->RemoveUser(this); }

  void set_called_computation(int index, HloComputation* computation) {
    called_computations_[index] = computation;
  }
  // Indices of computations in called_computations_ for instructions which call
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

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  virtual std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const {
    // TODO(b/80131774): This should be pure virtual.
    LOG(FATAL) << "Unimplemented method.";
  }

  // Implementation for non-common logic of ExtraAttributesToString.
  virtual std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const {
    return {};
  }

  // Implementation for IsElementwise if operand_idx is nullopt and for
  // IsElementwiseOnOperand if otherwise.
  //
  // NOTE: For all instructions other than kFusion, being elementwise on one of
  // the operands is equivalent to being elementwise on all the operands.
  virtual bool IsElementwiseImpl(
      const absl::optional<int64>& operand_idx) const;
  // Prints an instruction to a string.
  //
  // The canonical string representation needs to name operands and instruction
  // names in a consistent way. This is implemented through the
  // canonical_name_map.
  string ToStringWithCanonicalNameMap(
      const HloPrintOptions& options,
      CanonicalNameMap* canonical_name_map) const;

  // Prints an operand to a string.
  virtual string OperandsToStringWithCanonicalNameMap(
      const HloPrintOptions& options,
      CanonicalNameMap* canonical_name_map) const;

  // Allow HloInstruction to access the ToStringWithCanonicalNameMap() and
  // OperandsToStringWithCanonicalNameMap() functions.
  friend class HloComputation;

  // See comments on Identical().
  virtual bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const;

  // Creates an n-ary elementwise operation.
  static std::unique_ptr<HloInstruction> CreateNary(
      const Shape& shape, HloOpcode opcode,
      absl::Span<HloInstruction* const> operands);

  // Adds a user for this instruction.
  void AddUser(HloInstruction* user);

  // Removes a user for this instruction.
  void RemoveUser(HloInstruction* user);

  // Returns how this instruction uses elements of its `i`th operand.
  UseKind OperandElementUse(int64 i) const;

  // Helper for implementing backend_config().  Parses backend_config_ into the
  // given proto.
  Status GetBackendConfigInternal(tensorflow::protobuf::Message* proto) const;

  int unique_id_;  // Unique to this HloInstruction within a HloModule

  // Opcode for this instruction.
  HloOpcode opcode_;

  // Instruction operands.
  InstructionVector operands_;

  // The set of control predecessors of this instruction.
  std::vector<HloInstruction*> control_predecessors_;

  // The users of this instruction. Users are HLOs where this instruction is an
  // operand. The vector users_ and the set user_set_ contain identical
  // members. The set enables fast membership testing and the vector enables
  // fast, stable iteration.
  std::vector<HloInstruction*> users_;
  std::unordered_set<const HloInstruction*> user_set_;

  // The set of control successors of this instruction.
  std::vector<HloInstruction*> control_successors_;

  // The computation in which this instruction is contained.
  HloComputation* parent_ = nullptr;

  // Result shape of this instruction.
  Shape shape_;

  // Describes the dimension numbers used for a dot.
  std::unique_ptr<DotDimensionNumbers> dot_dimension_numbers_;

  // Used to tag kCopy instructions that are eligible for copy elision.
  bool copy_elision_allowed_ = true;

  // The sharding, if one exists.
  // Uses std::shared_ptr to allow reuse of the same sharding object between
  // HloInstructions and other components as HloSharding can be very large for
  // many element tuples.
  std::shared_ptr<const HloSharding> sharding_;

  // Fields used by the kDomain instruction.
  std::unique_ptr<DomainMetadata> operand_side_metadata_;
  std::unique_ptr<DomainMetadata> user_side_metadata_;

  // Computations called by this instruction.
  std::vector<HloComputation*> called_computations_;

  // A trace instruction that consumes this instruction.
  //
  // Invariant: if trace_instruction_ != nullptr, trace_instruction has this as
  // an operand.
  HloInstruction* trace_instruction_ = nullptr;

  // The backend-specific configuration for how a backend should compile this
  // HLO. See the documentation on backend_config().
  string backend_config_;

  // Information used to communicate to the implementation about the algorithm
  // used to produce results. See the documentation on precision_config().
  PrecisionConfigProto precision_config_;

  // String identifier for instruction.
  string name_;

  // Metadata for debugging.
  OpMetadata metadata_;

  // The number of partitions per outer dimension (listed in order from
  // outer-most dimension first).
  std::vector<int64> outer_dimension_partitions_;

  TF_DISALLOW_COPY_AND_ASSIGN(HloInstruction);
};

string ToString(HloInstruction::FusionKind kind);
StatusOr<HloInstruction::FusionKind> StringToFusionKind(
    const string& kind_name);

// Custom (de)stringification functions for protos that live inside
// HloInstruction.
string PaddingConfigToString(const PaddingConfig& padding);
string OpMetadataToString(const OpMetadata& metadata);
string RandomDistributionToString(const RandomDistribution& distribution);
string PrecisionToString(const PrecisionConfigProto::Precision& precision);
string ConvolutionDimensionNumbersToString(
    const ConvolutionDimensionNumbers& dnums);

StatusOr<RandomDistribution> StringToRandomDistribution(const string& name);
StatusOr<PrecisionConfigProto::Precision> StringToPrecision(const string& name);

std::ostream& operator<<(std::ostream& os, HloInstruction::FusionKind kind);

// Map classes that guarantee a deterministic iteration order when the key is
// an HloInstruction* or a const HloInstruction*.
// To make the iteration order over the map deterministic, the comparator
// should not be using the pointer values, but rather an intrinsic property of
// the hlo. Exception: null pointer values compare less than non-null.
//
// Note that this cannot be used for HLO instructions across multiple modules
// since the id of HLO instructions are only unique within each HLO module.
struct HloPtrComparator {
  bool operator()(const HloInstruction* const& lhs,
                  const HloInstruction* const& rhs) const {
    if (rhs == nullptr) {
      // Nothing compares less than nullptr.
      return false;
    }
    if (lhs == nullptr) {
      return true;
    }
    return lhs->unique_id() < rhs->unique_id();
  }
};

template <typename ValueT>
using HloInstructionMap = std::map<HloInstruction*, ValueT, HloPtrComparator>;

template <typename ValueT>
using ConstHloInstructionMap =
    std::map<const HloInstruction*, ValueT, HloPtrComparator>;

using HloInstructionSet = std::set<HloInstruction*, HloPtrComparator>;
using ConstHloInstructionSet =
    std::set<const HloInstruction*, HloPtrComparator>;

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTION_H_
