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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_IR_HLO_INSTRUCTION_H_
#define TENSORFLOW_COMPILER_XLA_HLO_IR_HLO_INSTRUCTION_H_

#include <functional>
#include <iosfwd>
#include <list>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_clone_context.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_domain_metadata.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "tensorflow/compiler/xla/iterator_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/printer.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/mapped_ptr_container_sorter.h"
#include "tensorflow/compiler/xla/service/name_uniquer.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/lib/gtl/iterator_range.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/protobuf.h"
#include "tensorflow/tsl/platform/status.h"

namespace xla {

class HloComputation;
class HloModule;

absl::string_view PrintName(absl::string_view name, bool print_ids);

// A bunch of switches that control how the hlo text should be printed.
class HloPrintOptions {
 public:
  enum class PrintSubcomputationMode {
    kOff,                  // Do not print anything about subcomputations.
    kNameOnly,             // Only print the name of subcomputations.
    kFullBodies,           // Print the full bodies of subcomputations.
    kNonSequentialBodies,  // Print the full bodies of subcomputations that are
                           // not in a sequential context.
  };

  // Constructs the default print options: don't print large constants, don't
  // compact operands, no indentation.
  HloPrintOptions()
      : print_large_constants_(false),
        print_only_essential_constants_(false),
        print_subcomputation_mode_(PrintSubcomputationMode::kNameOnly),
        print_metadata_(true),
        print_backend_config_(true),
        print_infeed_outfeed_config_(true),
        compact_operands_(false),
        include_layout_in_shapes_(true),
        print_result_shape_(true),
        print_operand_shape_(true),
        print_operand_names_(true),
        print_operand_index_annotation_interval_(5),
        print_program_shape_(true),
        print_percent_(true),
        print_control_dependencies_(true),
        canonicalize_instruction_names_(false),
        indent_amount_(0),
        is_in_nested_computation_(false),
        print_ids_(true),
        canonicalize_computations_(false),
        print_extra_attributes_(true),
        syntax_sugar_async_ops_(true) {}

  static HloPrintOptions ShortParsable() {
    return HloPrintOptions()
        .set_print_large_constants(true)
        .set_print_subcomputation_mode(PrintSubcomputationMode::kNameOnly)
        .set_print_metadata(false)
        .set_print_backend_config(false)
        .set_print_operand_shape(false)
        .set_print_operand_index_annotation_interval(0)
        .set_print_program_shape(false)
        .set_print_percent(false)
        .set_print_control_dependencies(false);
  }

  // Options to produce the canonical string representing an isomorphic
  // computation graph.
  static HloPrintOptions Canonical() {
    return HloPrintOptions()
        .set_print_subcomputation_mode(PrintSubcomputationMode::kFullBodies)
        .set_print_metadata(false)
        .set_print_backend_config(false)
        // Compact option won't print name of operand_k, where k > a threshold.
        // Canonical (and Fingerprint) must include all operands.
        .set_compact_operands(false)
        .set_print_operand_names(false)
        .set_print_operand_shape(true)
        // No index annotations as they are only for ease of human inspection.
        .set_print_operand_index_annotation_interval(0)
        .set_print_program_shape(false)
        .set_print_percent(false)
        .set_print_control_dependencies(false)
        .set_canonicalize_instruction_names(true);
  }

  // Options to produce a fingerprint of an HLO instruction.
  // Based on Canonical() with some important changes commented below.
  static HloPrintOptions Fingerprint() {
    return Canonical()
        // Exclude because they do not affect HLO optimizations.
        .set_print_infeed_outfeed_config(false)
        // Exclude floating point constant literals that are not all zeros, all
        // ones, or integers because they may be randomly initialized weights,
        // which may be changed across different runs.
        .set_print_only_essential_constants(true)
        // Remove "id" in "name.id" (after period) because it can be
        // non-deterministic. This mainly affects computations' names because
        // canonicalized instructions' names are in "tmp_id" format.
        .set_print_ids(false)
        // Sort computations.
        .set_canonicalize_computations(true);
  }

  // Options to produce a fingerprint of an HLO module and computation.
  // Shorter (and therefore faster) than Fingerprint().
  static HloPrintOptions ModuleFingerprint() {
    return Fingerprint()
        // Operand shapes can be inferred from output shapes and canonicalized
        // names when we have an entire computation.
        .set_print_operand_shape(false);
  }

  // If true, large constants will be printed out.
  HloPrintOptions& set_print_large_constants(bool value) {
    print_large_constants_ = value;
    return *this;
  }

  // If true, only integer, all-zero, are all-one constants will be printed out.
  HloPrintOptions& set_print_only_essential_constants(bool value) {
    print_only_essential_constants_ = value;
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

  // If true, infeed_config and outfeed_config will be printed.
  HloPrintOptions& set_print_infeed_outfeed_config(bool value) {
    print_infeed_outfeed_config_ = value;
    return *this;
  }

  // If true, result shapes will be printed.
  HloPrintOptions& set_print_result_shape(bool value) {
    print_result_shape_ = value;
    return *this;
  }

  // If true, operands' shapes will be printed.
  HloPrintOptions& set_print_operand_shape(bool value) {
    print_operand_shape_ = value;
    return *this;
  }

  // If true, operands' shapes will be printed.
  HloPrintOptions& set_print_operand_index_annotation_interval(int64_t value) {
    print_operand_index_annotation_interval_ = value;
    return *this;
  }

  // If true, the operand names will be printed.
  HloPrintOptions& set_print_operand_names(bool value) {
    print_operand_names_ = value;
    return *this;
  }

  // If true, all printed names include unique identifiers.
  HloPrintOptions& set_print_ids(bool value) {
    print_ids_ = value;
    return *this;
  }

  // If true, the HLO includes its attributes.
  HloPrintOptions& set_print_extra_attributes(bool value) {
    print_extra_attributes_ = value;
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

  // If true, control dependencies will be printed.
  HloPrintOptions& set_print_control_dependencies(bool value) {
    print_control_dependencies_ = value;
    return *this;
  }

  // If true, uses the async operation syntax sugar to print async-start,
  // async-update, and async-done HLOs. If the syntax sugar is enabled, the
  // computations called by these instructions will not be printed and instead
  // the root of the called computation will be printed instead of these
  // instructions and -start, -update, and -done suffixes will be appended to
  // the opcode of the async operation. For example, for an HLO module where the
  // syntax sugar is off:
  //
  // HloModule Module
  //
  // %AsyncOp (p0.1: f32[10]) -> f32[20] {
  //   %p0.1 = f32[10]{0} parameter(0)
  //   ROOT %custom-call = f32[20]{0} custom-call(f32[10]{0} %p0.1),
  //                                  custom_call_target="foo"
  // }
  //
  // ENTRY %Entry (p0: f32[10]) -> f32[20] {
  //   %p0 = f32[10]{0} parameter(0)
  //   %async-start = ((f32[10]{0}), f32[20]{0}, s32[]) async-start(%p0),
  //                                                    calls=%AsyncOp
  //   %async-update = ((f32[10]{0}), f32[20]{0}, s32[]) async-update(
  //                                                         %async-start),
  //                                                     calls=%AsyncOp
  //   ROOT %async-done = f32[20]{0} async-done(%async-update), calls=%AsyncOp
  // }
  //
  // will be printed as following if the syntax sugar is enabled:
  //
  // HloModule Module
  //
  // ENTRY %Entry (p0: f32[10]) -> f32[20] {
  //   %p0 = f32[10]{0} parameter(0)
  //   %async-start = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-start(%p0),
  //                                                    custom_call_target="foo"
  //   %async-update = ((f32[10]{0}), f32[20]{0}, s32[]) custom-call-update(
  //                                                         %async-start),
  //                                                    custom_call_target="foo"
  //   ROOT %async-done = f32[20]{0} custom-call-done(%async-update),
  //                                                    custom_call_target="foo"
  // }
  HloPrintOptions& set_syntax_sugar_async_ops(bool value) {
    syntax_sugar_async_ops_ = value;
    return *this;
  }

  // If true, only a part of operands will be printed out (note that in this
  // case the text will not be parsable).
  HloPrintOptions& set_compact_operands(bool value) {
    compact_operands_ = value;
    return *this;
  }

  // If true, include the layout in any shapes that are printed (instruction
  // and operands).
  HloPrintOptions& set_include_layout_in_shapes(bool value) {
    include_layout_in_shapes_ = value;
    return *this;
  }

  // If true, canonicalizes instructions' name. Instead of using "%foo.1" as
  // the name of an instruction, we use "%tmp_1", "%tmp_2" etc.
  HloPrintOptions& set_canonicalize_instruction_names(bool value) {
    canonicalize_instruction_names_ = value;
    return *this;
  }

  // If true, canonicalizes computations, sorting by computations' names.
  HloPrintOptions& set_canonicalize_computations(bool value) {
    canonicalize_computations_ = value;
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
  bool print_only_essential_constants() const {
    return print_only_essential_constants_;
  }
  PrintSubcomputationMode print_subcomputation_mode() const {
    return print_subcomputation_mode_;
  }
  bool print_metadata() const { return print_metadata_; }
  bool print_backend_config() const { return print_backend_config_; }
  bool print_infeed_outfeed_config() const {
    return print_infeed_outfeed_config_;
  }
  bool compact_operands() const { return compact_operands_; }
  bool include_layout_in_shapes() const { return include_layout_in_shapes_; }
  bool print_result_shape() const { return print_result_shape_; }
  bool print_operand_shape() const { return print_operand_shape_; }
  bool print_operand_names() const { return print_operand_names_; }
  int64_t print_operand_index_annotation_interval() const {
    return print_operand_index_annotation_interval_;
  }
  bool print_ids() const { return print_ids_; }
  bool print_program_shape() const { return print_program_shape_; }
  bool print_percent() const { return print_percent_; }
  bool print_control_dependencies() const {
    return print_control_dependencies_;
  }
  bool print_extra_attributes() const { return print_extra_attributes_; }
  bool syntax_sugar_async_ops() const { return syntax_sugar_async_ops_; }
  bool canonicalize_instruction_names() const {
    return canonicalize_instruction_names_;
  }
  bool canonicalize_computations() const { return canonicalize_computations_; }
  int indent_amount() const { return indent_amount_; }
  int is_in_nested_computation() const { return is_in_nested_computation_; }

 private:
  bool print_large_constants_;
  bool print_only_essential_constants_;
  PrintSubcomputationMode print_subcomputation_mode_;
  bool print_metadata_;
  bool print_backend_config_;
  bool print_infeed_outfeed_config_;
  bool compact_operands_;
  bool include_layout_in_shapes_;
  bool print_result_shape_;
  bool print_operand_shape_;
  bool print_operand_names_;
  // The interval between the /*index=*/ annotated operands. 0 means never print
  // the annotation, 1 means print annotation for every operand.
  int64_t print_operand_index_annotation_interval_;
  bool print_program_shape_;
  bool print_percent_;
  bool print_control_dependencies_;
  bool canonicalize_instruction_names_;
  int indent_amount_;
  bool is_in_nested_computation_;
  bool print_ids_;
  bool canonicalize_computations_;
  bool print_extra_attributes_;
  bool syntax_sugar_async_ops_;
};

// For canonical string output, we need to have a canonical way to rename
// each instruction and its operands. Each operand is renamed as "tmp_<xxx>",
// where <xxx> is an index starting from 0.
class CanonicalNameMap {
 public:
  const std::string& LookupOrInsert(int unique_id) {
    std::string& canonical_name = canonical_name_map_[unique_id];
    if (canonical_name.empty()) {
      absl::StrAppend(&canonical_name, "tmp_", canonical_name_map_.size() - 1);
    }
    return canonical_name;
  }

  void Reserve(size_t size) { canonical_name_map_.reserve(size); }

 private:
  absl::flat_hash_map<int, std::string> canonical_name_map_;
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

  inline static constexpr char kMainExecutionThread[] = "main";

  virtual ~HloInstruction() { DetachFromOperandsAndUsers(); }

  // Detaches an instruction from its operands and users. That is, remove the
  // instruction from each operand's user set and user's operand set.
  void DetachFromOperandsAndUsers();

  // Adds a derived instruciton to the parent compuation of this instruction.
  // Also update setup the new instruction as a derived instruction.
  HloInstruction* AddInstruction(
      std::unique_ptr<HloInstruction> derived_instruction);

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
      const absl::flat_hash_map<int64_t, HloInstruction*>& instruction_map,
      const absl::flat_hash_map<int64_t, HloComputation*>& computation_map = {},
      bool prohibit_empty_literal = true);

  // Creates a parameter-retrieving instruction.
  static std::unique_ptr<HloInstruction> CreateParameter(
      int64_t parameter_number, const Shape& shape, const std::string& name);

  // Creates a literal constant instruction.
  static std::unique_ptr<HloInstruction> CreateConstant(Literal literal);

  // Creates an Iota instruction.
  static std::unique_ptr<HloInstruction> CreateIota(const Shape& shape,
                                                    int64_t iota_dimension);

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
      std::optional<int64_t> async_group_id = std::nullopt,
      absl::string_view async_execution_thread = kMainExecutionThread);
  static std::unique_ptr<HloInstruction> CreateAsyncUpdate(
      const Shape& shape, HloInstruction* operand,
      HloComputation* async_computation,
      std::optional<int64_t> async_group_id = std::nullopt,
      absl::string_view async_execution_thread = kMainExecutionThread);
  static std::unique_ptr<HloInstruction> CreateAsyncDone(
      const Shape& shape, HloInstruction* operand,
      HloComputation* async_computation,
      std::optional<int64_t> async_group_id = std::nullopt,
      absl::string_view async_execution_thread = kMainExecutionThread);

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
  // dimensions specified in 'dimension_numbers'.
  static std::unique_ptr<HloInstruction> CreateDot(
      const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
      const DotDimensionNumbers& dimension_numbers,
      const PrecisionConfig& precision_config);

  // Creates a reduce-precision op, where operand is the data to reduce in
  // precision, and exponent_bits and mantissa_bits describe the precision to
  // reduce it to.
  static std::unique_ptr<HloInstruction> CreateReducePrecision(
      const Shape& shape, HloInstruction* operand, const int exponent_bits,
      const int mantissa_bits);

  // Creates an all-gather op, which concats the operands of all participants
  // along all_gather_dimension. The replica_groups, channel_id, and
  // use_global_device_ids arguments are identical to those in all-reduce,
  // except that the order of the group members determines the concatenation
  // order of inputs from different participants.
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
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids);

  // Creates a reduce-scatter operation which reduces its inputs across the
  // given replica groups and then scatters the reduced data across the N
  // participants.
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
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id,
      const std::optional<int64_t>& split_dimension = std::nullopt);

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
      HloInstruction* operand, HloInstruction* token, int64_t channel_id,
      bool is_host_transfer = false);

  // Blocks until data transfer for the Send instruction (operand) is complete.
  // The operand must be kSend.
  static std::unique_ptr<HloInstruction> CreateSendDone(
      HloInstruction* operand, bool is_host_transfer = false);

  // Creates an asynchronous receive instruction with the given channel id,
  // which allocates resources to receive data of the given shape from a unique
  // send instruction in another computation that has the same channel id.  If
  // is_host_transfer is true, then this Recv operation transfers data from the
  // host.
  static std::unique_ptr<HloInstruction> CreateRecv(
      const Shape& shape, HloInstruction* token, int64_t channel_id,
      bool is_host_transfer = false);

  // Blocks until data transfer for the Recv instruction (operand) is complete
  // and returns the receive buffer. The operand must be kRecv.
  static std::unique_ptr<HloInstruction> CreateRecvDone(
      HloInstruction* operand, bool is_host_transfer = false);

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
  // Precondition: dim_sizes.size() == shape.rank()
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
      const Shape& shape, FusionKind fusion_kind, HloInstruction* fused_root);

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
  // all be tokens, and there must be at least one operand.
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
  bool IsRoot() const;

  // Does this instruction have no users.
  bool IsDead() const { return users_.empty() && !IsRoot(); }

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

  // Returns the index of 'target' in the operands sequence.
  // Precondition: target must be an operand (or a fatal error will occur).
  int64_t operand_index(const HloInstruction* target) const;

  // Returns the number of users of this instruction.
  int64_t user_count() const { return users_.size(); }

  // Returns the users of this instruction.
  const std::vector<HloInstruction*>& users() const { return users_; }

  // Returns the index of the user in the users() vector.
  //
  // Precondition: `user` is a user of the instruction.
  int64_t UserId(HloInstruction* user);

  // Returns true if this instruction is a user of 'instruction'.
  bool IsUserOf(const HloInstruction* instruction) const {
    return ContainsKey(instruction->user_map_, this);
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

  // Drops all control predecessors and successors from this HLO instruction,
  // and the maintain the transitivie control dependencies between
  // control predecessors and control successors.
  Status SafelyDropAllControlDependencies();

  // Returns if instruction has any control dependencies.
  bool HasControlDependencies() const;

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

  // Generates a hash value of an HLO instruction. Hash considers
  // information on opcode, shape, operands, and typically a root instruction.
  // This function returns the same hash value for equivalent HLO instructions,
  // with respect to HloInstruction::Identical() method.
  // TODO(majnemer): Make the comment here more crisp & accurate.
  template <typename H>
  friend H AbslHashValue(H h, const HloInstruction& hlo) {
    h = H::combine(std::move(h), hlo.opcode(), hlo.shape());

    if (!hlo.IsCrossModuleAllReduce()) {
      for (size_t i = 0; i < hlo.operands().size(); ++i) {
        h = H::combine(std::move(h), hlo.operand(i)->shape());
      }
      h = H::combine(std::move(h), hlo.operand_count());
    }

    if (hlo.opcode() == HloOpcode::kFusion) {
      h = H::combine(std::move(h), *hlo.fused_expression_root(),
                     hlo.fusion_kind(), hlo.fused_instruction_count(),
                     hlo.fused_parameters().size());
    }
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
  Status ReplaceUseWith(HloInstruction* user, HloInstruction* new_producer);

  // Same as ReplaceUseWith(), but new_producer can have a different shape.
  Status ReplaceUseWithDifferentShape(HloInstruction* user,
                                      HloInstruction* new_producer);

  // Same as ReplaceUseWith but only replaces the use at the given operand
  // number.
  Status ReplaceUseWith(HloInstruction* user, int operand_number,
                        HloInstruction* new_producer);
  Status ReplaceUseWithDifferentShape(HloInstruction* user, int operand_number,
                                      HloInstruction* new_producer);

  // Replaces the specified operand with new_operand. The old and new operands
  // must have compatible shapes ignoring floating-point precision.
  //
  // This function does NOT remove duplicated operands even if this instruction
  // is a fusion, so that the existing operand numbers do not change.
  Status ReplaceOperandWith(int64_t operand_num, HloInstruction* new_operand);

  // Same as ReplaceOperandWith(), but new_operand can have a different shape.
  Status ReplaceOperandWithDifferentShape(int64_t operand_num,
                                          HloInstruction* new_operand);

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
  Status ReplaceAllUsesWith(HloInstruction* new_producer);

  // Same as ReplaceAllUsesWith, but new_producer can have a different shape.
  Status ReplaceAllUsesWithDifferentShape(HloInstruction* new_producer);

  // Same as ReplaceAllUsesWith, but only replace given set of users.
  Status ReplaceUsesWith(absl::Span<HloInstruction* const> users,
                         HloInstruction* new_producer);
  Status ReplaceAllUsesWithDifferentShape(
      absl::Span<HloInstruction* const> users, HloInstruction* new_producer);

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
      absl::FunctionRef<bool(const HloInstruction*, const HloInstruction*)>;
  Status AcceptWithOperandOrder(DfsHloVisitor* visitor,
                                CompareFunction operand_order,
                                bool call_finish_visit = true);

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

  // Returns true whether this instruction is effectively a bitcast. Currently,
  // this means it either is a bitcast, or it is a transpose that is effectively
  // a bitcast.
  bool IsEffectiveBitcast() const;

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
  const std::vector<HloComputation*>& branch_computations() const;
  int branch_count() const;
  HloComputation* branch_computation(int b) const;
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
    return Print(printer, HloPrintOptions());
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
  std::string ToString() const { return ToString(HloPrintOptions()); }
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
  void set_sharding(const HloSharding& sharding) {
    set_sharding(std::make_shared<const HloSharding>(sharding));
  }
  void set_sharding(std::shared_ptr<const HloSharding> sharding) {
    sharding_ = std::move(sharding);
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

  // Returns the computations this instruction directly calls (if any).
  const std::vector<HloComputation*>& called_computations() const {
    return called_computations_;
  }

  // Replaces all called computations based on a map function. This is needed
  // when we clone hlo_computations and want to let the instructions to point
  // to the newly cloned nodes.
  void ReplaceCalledComputations(
      absl::FunctionRef<HloComputation*(HloComputation*)> map_function) {
    for (int64_t i = 0; i < called_computations_.size(); ++i) {
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
  virtual void ClearCalledComputations() { called_computations_.clear(); }

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
  const std::string& name() const { return name_; }

  // Sets the string identifier for this instruction. Name will be sanitized to
  // match the regexp "[a-zA-Z_][a-zA-Z0-9_.-]*".
  //
  // See also HloModule::SetAndUniquifyInstrName(), which does this plus
  // UniqufyName().
  void SetAndSanitizeName(absl::string_view name) {
    name_ = NameUniquer::GetSanitizedName(name);
  }

  // Use the given NameUniquer to select a unique name for the instruction based
  // on the instruction's existing name.
  //
  // See also HloModule::SetAndUniquifyInstrName(), which does this plus
  // SetAndSanitizeName().
  void UniquifyName(NameUniquer* name_uniquer);

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
  StatusOr<ConfigProto> backend_config() const {
    ConfigProto proto;
    TF_RETURN_IF_ERROR(GetBackendConfigInternal(&proto));
    return std::move(proto);
  }

  Status set_backend_config(const tsl::protobuf::Message& proto) {
    backend_config_ = proto;
    return OkStatus();
  }

  bool has_backend_config() const { return !backend_config_.empty(); }

  void clear_backend_config() { backend_config_.clear(); }

  void CopyBackendConfigFrom(const HloInstruction* other) {
    backend_config_ = other->backend_config_.Clone();
  }

  void set_frontend_attributes(FrontendAttributes frontend_attributes) {
    frontend_attributes_ = std::move(frontend_attributes);
  }

  void add_frontend_attributes(FrontendAttributes frontend_attributes) {
    frontend_attributes_.mutable_map()->insert(
        frontend_attributes.map().begin(), frontend_attributes.map().end());
  }

  const FrontendAttributes& frontend_attributes() const {
    return frontend_attributes_;
  }

  // Getter/setter for raw JSON-encoded backend config.  Prefer the
  // functions above that deal in proto Messages where possible.
  const std::string& raw_backend_config_string() const {
    return backend_config_.GetRawString();
  }
  void set_raw_backend_config_string(std::string config_str) {
    backend_config_ = std::move(config_str);
  }

  bool is_default_config() const { return is_default_config_; }
  void set_default_config() { is_default_config_ = true; }

  // Returns a string representation of a proto in the format used by
  // raw_backend_config_string.
  //
  // This is morally equivalent to:
  //
  //   HloInstruction instr;
  //   TF_RETURN_IF_ERROR(instr.set_backend_config(proto));
  //   return instr.raw_backend_config_string();
  //
  static StatusOr<std::string> BackendConfigToRawString(
      const tsl::protobuf::Message& proto);

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

  // Sets the debug metadata for this instruction, excluding creation_pass_id,
  // which should never be copied anywhere.
  void set_metadata(const OpMetadata& metadata) {
    int64_t creation_pass_id = metadata_.creation_pass_id();
    metadata_ = metadata;
    metadata_.set_creation_pass_id(creation_pass_id);
  }

  void set_size_of_generated_code_in_bytes(int64_t code_size_in_bytes) {
    metadata_.set_size_of_generated_code_in_bytes(code_size_in_bytes);
  }
  void set_size_of_memory_working_set_in_bytes(
      int64_t working_set_size_in_bytes) {
    metadata_.set_size_of_memory_working_set_in_bytes(
        working_set_size_in_bytes);
  }
  void set_creation_pass_id(int64_t pass_id) {
    metadata_.set_creation_pass_id(pass_id);
  }
  void set_metadata_op_name(const std::string& name) {
    metadata_.set_op_name(name);
  }
  void set_logical_creation_pass_id(int64_t pass_id) {
    metadata_.set_logical_creation_pass_id(pass_id);
  }
  void set_metadata_deduplicated_name(const std::string& deduplicated_name) {
    metadata_.set_deduplicated_name(deduplicated_name);
  }
  const OpMetadata& metadata() const { return metadata_; }

  // Set/get the computation containing this instruction. set_parent should only
  // be called by HloComputation methods which add/remove instructions to
  // computations.
  void set_parent(HloComputation* computation) { parent_ = computation; }
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
  const tsl::gtl::iterator_range<UnwrappingIterator<
      std::list<std::unique_ptr<HloInstruction>>::const_iterator>>
  fused_instructions() const;

  const tsl::gtl::iterator_range<
      UnwrappingIterator<std::list<std::unique_ptr<HloInstruction>>::iterator>>
  fused_instructions();

  // Delegates to HloFusionInstruction::fused_instruction_count.
  int64_t fused_instruction_count() const;

  // Delegates to HloFusionInstruction::fused_parameter.
  HloInstruction* fused_parameter(int64_t parameter_number) const;

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
  const std::vector<ReplicaGroup>& replica_groups() const;

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

  // Delegates to HloDomainInstruction::operand_side_metadata().
  const DomainMetadata& operand_side_metadata() const;

  // Delegates to HloDomainInstruction::user_side_metadata().
  const DomainMetadata& user_side_metadata() const;

  // Returns true if the instruction is an async-start, async-update, or
  // async-done.
  bool IsAsynchronous() const;

  // Returns the computation that will executed asynchronously.
  HloComputation* async_wrapped_computation() const;

  // Delagates to HloAsyncInstruction::async_wrapped_instruction().
  HloInstruction* async_wrapped_instruction() const;

  // Delagates to HloAsyncInstruction::async_wrapped_opcode().
  HloOpcode async_wrapped_opcode() const;

  // Delegates to HloAsyncInstruction::async_group_id().
  std::optional<int64_t> async_group_id() const;

  // Delegates to HloAsyncInstruction::set_async_group_id().
  void set_async_group_id(std::optional<int64_t> async_group_id);

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

  // Appends operand to the list of operands and adds this instruction as a user
  // of the operand.
  void AppendOperand(HloInstruction* operand);

  // Old methods kept for smooth subclassing transition END.

  HloInstruction(const HloInstruction&) = delete;
  HloInstruction& operator=(const HloInstruction&) = delete;

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
  friend class HloComputation;
  // Wrapper class of string format and protobuf format of BackendConfig.
  class BackendConfigRep {
   public:
    const tsl::protobuf::Message* GetProtoPtr() const { return proto_.get(); }

    const std::string& GetRawString() const;

    BackendConfigRep Clone() const;

    bool operator==(const BackendConfigRep& other) const;
    bool operator!=(const BackendConfigRep& other) const {
      return !(*this == other);
    }

    bool empty() const { return proto_ == nullptr && raw_string_.empty(); }

    void clear() {
      proto_.reset();
      raw_string_.clear();
    }

    BackendConfigRep& operator=(std::string raw_string);
    BackendConfigRep& operator=(const tsl::protobuf::Message& proto);
    void SetProto(const tsl::protobuf::Message& proto);

   private:
    std::unique_ptr<tsl::protobuf::Message> proto_;
    // If proto_ is not null, raw_string_ is a lazy cache of its string format.
    mutable std::string raw_string_;
  };

  bool IdenticalInternal(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloInstruction*, const HloInstruction*)>
          eq_operands,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations,
      bool layout_sensitive, bool sharding_sensitive,
      bool ignore_channel_id_values,
      bool ignore_commutative_operand_order) const;

  // Implementation for non-common logic of CloneWithNewOperands.
  virtual std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const {
    // TODO(b/80131774): This should be pure virtual.
    LOG(FATAL) << "Unimplemented method.";
  }

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
  void AddUser(HloInstruction* user);

  // Removes a user for this instruction.
  void RemoveUser(HloInstruction* user);

  // Helper for implementing backend_config().  Parses backend_config_ into the
  // given proto.
  Status GetBackendConfigInternal(tsl::protobuf::Message* proto) const;

  // Mark this instruction as dead. Accessed by friend class HloInstruction.
  void MarkAsDead() { marked_as_dead_ = true; }

  // Has this instruction been marked as dead? Accessed by friend class
  // HloInstruction.
  bool IsMarkedAsDead() const { return marked_as_dead_; }

  int unique_id_;  // Unique to this HloInstruction within a HloModule

  // Opcode for this instruction.
  HloOpcode opcode_;

  // Instruction operands.
  InstructionVector operands_;

  // The set of control predecessors of this instruction.
  // Note that the order of the instructions in the vector influences the order
  // computed in HloComputation::ComputeInstructionPostOrder, which may
  // influence the result of the compilation by changing the scheduling. We are
  // not sure if it matters.
  std::vector<HloInstruction*> control_predecessors_;

  // The users of this instruction. Users are HLOs where this instruction is an
  // operand. The vector users_ and the map user_map_ contain identical members.
  // The map enables fast membership testing and the vector enables fast, stable
  // iteration. The value in the map contains the index of the instruction in
  // the vector what enables fast removal.
  std::vector<HloInstruction*> users_;
  absl::flat_hash_map<const HloInstruction*, int64_t> user_map_;

  // The set of control successors of this instruction.
  std::vector<HloInstruction*> control_successors_;

  // The computation in which this instruction is contained.
  HloComputation* parent_ = nullptr;

  // Result shape of this instruction.
  Shape shape_;

  // The sharding, if one exists.
  // Uses std::shared_ptr to allow reuse of the same sharding object between
  // HloInstructions and other components as HloSharding can be very large for
  // many element tuples.
  std::shared_ptr<const HloSharding> sharding_;

  // Computations called by this instruction.
  std::vector<HloComputation*> called_computations_;

  // The backend-specific configuration for how a backend should compile this
  // HLO. See the documentation on backend_config().
  mutable BackendConfigRep backend_config_;

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
  FrontendAttributes frontend_attributes_;

  // String identifier for instruction.
  std::string name_;

  // Metadata for debugging.
  OpMetadata metadata_;

  // This field is assigned to true when backend_config_ is assigned to
  // a default configuration.
  bool is_default_config_ = false;

  // True if this instruction has already been detached from its user and
  // operands.
  bool cleaned_up_ = false;

  // Intrusive flag used by HloComputation, whether this instruction has
  // been marked as dead.
  bool marked_as_dead_;
};

// Explicit instantiations in hlo_instruction.cc.
extern template Status HloInstruction::Accept(DfsHloVisitor*, bool, bool);
extern template Status HloInstruction::Accept(ConstDfsHloVisitor*, bool, bool);
extern template Status HloInstruction::Visit(DfsHloVisitor* visitor);
extern template Status HloInstruction::Visit(ConstDfsHloVisitor* visitor);

absl::string_view ToString(HloInstruction::FusionKind kind);
StatusOr<HloInstruction::FusionKind> StringToFusionKind(
    absl::string_view kind_name);

// Custom (de)stringification functions for protos that live inside
// HloInstruction.
std::string PaddingConfigToString(const PaddingConfig& padding);
std::string FrontendAttributesToString(
    const FrontendAttributes& frontend_attributes);
std::string RandomAlgorithmToString(const RandomAlgorithm& algorithm);
std::string RandomDistributionToString(const RandomDistribution& distribution);
std::string PrecisionToString(const PrecisionConfig::Precision& precision);
std::string DotDimensionNumbersToString(const DotDimensionNumbers& dnums);
std::string ConvolutionDimensionNumbersToString(
    const ConvolutionDimensionNumbers& dnums);
std::string ReplicaGroupsToString(
    absl::Span<const ReplicaGroup> replica_groups);

StatusOr<RandomAlgorithm> StringToRandomAlgorithm(const std::string& name);
StatusOr<RandomDistribution> StringToRandomDistribution(
    const std::string& name);
StatusOr<PrecisionConfig::Precision> StringToPrecision(const std::string& name);
StatusOr<CustomCallSchedule> StringToCustomCallSchedule(absl::string_view name);
StatusOr<CustomCallApiVersion> StringToCustomCallApiVersion(
    absl::string_view name);

std::ostream& operator<<(std::ostream& os, HloInstruction::FusionKind kind);

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

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_HLO_IR_HLO_INSTRUCTION_H_
