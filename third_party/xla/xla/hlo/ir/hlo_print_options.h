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

#ifndef XLA_HLO_IR_HLO_PRINT_OPTIONS_H_
#define XLA_HLO_IR_HLO_PRINT_OPTIONS_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace xla {

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
  constexpr HloPrintOptions()
      : print_operand_index_annotation_interval_(5),
        print_subcomputation_mode_(PrintSubcomputationMode::kNameOnly),
        indent_amount_(0),
        print_large_constants_(false),
        print_only_essential_constants_(false),
        print_original_value_(true),
        print_metadata_(true),
        print_metadata_only_op_name_(false),
        print_backend_config_(true),
        sort_backend_config_(false),
        print_infeed_outfeed_config_(true),
        compact_operands_(false),
        include_layout_in_shapes_(true),
        print_result_shape_(true),
        print_operand_shape_(true),
        print_operand_names_(true),
        print_program_shape_(true),
        print_percent_(true),
        print_control_dependencies_(true),
        canonicalize_instruction_names_(false),
        is_in_nested_computation_(false),
        print_ids_(true),
        canonicalize_computations_(false),
        print_extra_attributes_(true),
        syntax_sugar_async_ops_(true),
        print_name_after_closing_brace_(false),
        print_full_replica_group_list_(false),
        print_parameter_number_(true) {}
  // Static reference to a default construction HloPrintOptions, to avoid
  // constructing a new one each time default is needed.
  static const HloPrintOptions& Default() {
    ABSL_CONST_INIT static const HloPrintOptions options;
    return options;
  }

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
        // Exclude floating point constant literals that are not all
        // zeros, all ones, or integers because they may be randomly
        // initialized weights, which may be changed across different
        // runs.
        .set_print_only_essential_constants(true)
        // Remove "id" in "name.id" (after period) because it can be
        // non-deterministic. This mainly affects computations' names
        // because canonicalized instructions' names are in "tmp_id"
        // format.
        .set_print_ids(false)
        // Sort computations.
        .set_canonicalize_computations(true)
        // Force to print full replica group list to avoid non-determinism.
        // With this flag set to false, the replica group list may be printed
        // in a compact form when iota_replica_group_list is present, which may
        // be different across different runs.
        .set_print_full_replica_group_list(true);
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

  // If true, origin will be printed.
  HloPrintOptions& set_print_original_value(bool value) {
    print_original_value_ = value;
    return *this;
  }
  // If true, metadata will be printed.
  HloPrintOptions& set_print_metadata(bool value) {
    print_metadata_ = value;
    return *this;
  }

  // If true and print_metadata is true, metadata op name will be printed. Other
  // metadata values will be omitted.
  HloPrintOptions& set_print_metadata_only_op_name(bool value) {
    print_metadata_only_op_name_ = value;
    return *this;
  }

  // If true, backend_config will be printed.
  HloPrintOptions& set_print_backend_config(bool value) {
    print_backend_config_ = value;
    return *this;
  }

  // If true, will attempt to sort the backend config's json representation
  // before printing it. If the backend config is a raw string that is not json,
  // it will be printed as is, without sorting.
  HloPrintOptions& set_sort_backend_config(bool value) {
    sort_backend_config_ = value;
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

  HloPrintOptions& set_print_full_replica_group_list(bool value) {
    print_full_replica_group_list_ = value;
    return *this;
  }

  // If true, uses the async operation syntax sugar to print async-start,
  // async-update, and async-done HLOs. If the syntax sugar is enabled and the
  // async computation is trivial (i.e. only a single instruction taking
  // computation parameters as operands, and otherwise is illegal), the
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

  HloPrintOptions& set_print_name_after_closing_brace(bool value) {
    print_name_after_closing_brace_ = value;
    return *this;
  }

  // If true, prints the parameter number of a parameter instruction.
  HloPrintOptions& set_print_parameter_number(bool value) {
    print_parameter_number_ = value;
    return *this;
  }

  bool print_large_constants() const { return print_large_constants_; }
  bool print_only_essential_constants() const {
    return print_only_essential_constants_;
  }
  PrintSubcomputationMode print_subcomputation_mode() const {
    return print_subcomputation_mode_;
  }
  bool print_original_value() const { return print_original_value_; }
  bool print_metadata() const { return print_metadata_; }
  bool print_metadata_only_op_name() const {
    return print_metadata_only_op_name_;
  }
  bool print_backend_config() const { return print_backend_config_; }
  bool sort_backend_config() const { return sort_backend_config_; }
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
  int print_name_after_closing_brace() const {
    return print_name_after_closing_brace_;
  }
  bool print_full_replica_group_list() const {
    return print_full_replica_group_list_;
  }
  bool print_parameter_number() const { return print_parameter_number_; }

 private:
  // The interval between the /*index=*/ annotated operands. 0 means never print
  // the annotation, 1 means print annotation for every operand.
  int64_t print_operand_index_annotation_interval_;
  PrintSubcomputationMode print_subcomputation_mode_;
  int indent_amount_;
  bool print_large_constants_;
  bool print_only_essential_constants_;
  bool print_original_value_;
  bool print_metadata_;
  bool print_metadata_only_op_name_;
  bool print_backend_config_;
  bool sort_backend_config_;
  bool print_infeed_outfeed_config_;
  bool compact_operands_;
  bool include_layout_in_shapes_;
  bool print_result_shape_;
  bool print_operand_shape_;
  bool print_operand_names_;
  bool print_program_shape_;
  bool print_percent_;
  bool print_control_dependencies_;
  bool canonicalize_instruction_names_;
  bool is_in_nested_computation_;
  bool print_ids_;
  bool canonicalize_computations_;
  bool print_extra_attributes_;
  bool syntax_sugar_async_ops_;
  bool print_name_after_closing_brace_;
  bool print_full_replica_group_list_;
  bool print_parameter_number_;
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

}  // namespace xla

#endif  // XLA_HLO_IR_HLO_PRINT_OPTIONS_H_
