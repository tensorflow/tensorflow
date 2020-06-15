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

#include "tensorflow/compiler/xla/service/hlo_parser.h"

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_domain_metadata.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_lexer.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_metadata.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {

namespace {

using absl::nullopt;
using absl::optional;
using absl::StrAppend;
using absl::StrCat;
using absl::StrFormat;
using absl::StrJoin;

// Creates and returns a schedule created using the order of the instructions in
// the HloComputation::instructions() vectors in the module.
HloSchedule ScheduleFromInstructionOrder(HloModule* module) {
  HloSchedule schedule(module);
  for (HloComputation* computation : module->computations()) {
    if (!computation->IsFusionComputation()) {
      for (HloInstruction* instruction : computation->instructions()) {
        schedule.GetOrCreateSequence(computation).push_back(instruction);
      }
    }
  }
  return schedule;
}

// Parser for the HloModule::ToString() format text.
class HloParserImpl : public HloParser {
 public:
  using LocTy = HloLexer::LocTy;

  explicit HloParserImpl(absl::string_view str) : lexer_(str) {}

  // Runs the parser and constructs the resulting HLO in the given (empty)
  // HloModule. Returns the error status in case an error occurred.
  Status Run(HloModule* module) override;

  // Returns the error information.
  std::string GetError() const { return StrJoin(error_, "\n"); }

  // Stand alone parsing utils for various aggregate data types.
  StatusOr<Shape> ParseShapeOnly();
  StatusOr<HloSharding> ParseShardingOnly();
  StatusOr<FrontendAttributes> ParseFrontendAttributesOnly();
  StatusOr<std::vector<bool>> ParseParameterReplicationOnly();
  StatusOr<Window> ParseWindowOnly();
  StatusOr<ConvolutionDimensionNumbers> ParseConvolutionDimensionNumbersOnly();
  StatusOr<PaddingConfig> ParsePaddingConfigOnly();
  StatusOr<std::vector<ReplicaGroup>> ParseReplicaGroupsOnly();

 private:
  using InstrNameTable =
      absl::flat_hash_map<std::string, std::pair<HloInstruction*, LocTy>>;

  // Returns the map from the instruction name to the instruction itself and its
  // location in the current scope.
  InstrNameTable& current_name_table() { return scoped_name_tables_.back(); }

  // Locates an instruction with the given name in the current_name_table() or
  // returns nullptr.
  //
  // When the name is not found or name is empty, if create_missing_instruction_
  // hook is registered and a "shape" is provided, the hook will be called to
  // create an instruction. This is useful when we reify parameters as they're
  // resolved; i.e. for ParseSingleInstruction.
  std::pair<HloInstruction*, LocTy>* FindInstruction(
      const std::string& name, const optional<Shape>& shape = nullopt);

  // Parse a single instruction worth of text.
  bool ParseSingleInstruction(HloModule* module);

  // Parses a module, returning false if an error occurred.
  bool ParseHloModule(HloModule* module);

  bool ParseComputations(HloModule* module);
  bool ParseComputation(HloComputation** entry_computation);
  bool ParseInstructionList(HloComputation** computation,
                            const std::string& computation_name);
  bool ParseInstruction(HloComputation::Builder* builder,
                        std::string* root_name);
  bool ParseInstructionRhs(HloComputation::Builder* builder,
                           const std::string& name, LocTy name_loc);
  bool ParseControlPredecessors(HloInstruction* instruction);
  bool ParseLiteral(Literal* literal, const Shape& shape);
  bool ParseTupleLiteral(Literal* literal, const Shape& shape);
  bool ParseNonTupleLiteral(Literal* literal, const Shape& shape);
  bool ParseDenseLiteral(Literal* literal, const Shape& shape);

  // Sets the sub-value of literal at the given linear index to the
  // given value. If the literal is dense, it must have the default layout.
  //
  // `loc` should be the source location of the value.
  bool SetValueInLiteral(LocTy loc, int64 value, int64 index, Literal* literal);
  bool SetValueInLiteral(LocTy loc, double value, int64 index,
                         Literal* literal);
  bool SetValueInLiteral(LocTy loc, bool value, int64 index, Literal* literal);
  bool SetValueInLiteral(LocTy loc, std::complex<double> value, int64 index,
                         Literal* literal);
  // `loc` should be the source location of the value.
  template <typename LiteralNativeT, typename ParsedElemT>
  bool SetValueInLiteralHelper(LocTy loc, ParsedElemT value, int64 index,
                               Literal* literal);

  // Checks whether the given value is within the range of LiteralNativeT.
  // `loc` should be the source location of the value.
  template <typename LiteralNativeT, typename ParsedElemT>
  bool CheckParsedValueIsInRange(LocTy loc, ParsedElemT value);
  template <typename LiteralNativeT>
  bool CheckParsedValueIsInRange(LocTy loc, std::complex<double> value);

  bool ParseOperands(std::vector<HloInstruction*>* operands);
  // Fills parsed operands into 'operands' and expects a certain number of
  // operands.
  bool ParseOperands(std::vector<HloInstruction*>* operands,
                     const int expected_size);

  // Describes the start, limit, and stride on every dimension of the operand
  // being sliced.
  struct SliceRanges {
    std::vector<int64> starts;
    std::vector<int64> limits;
    std::vector<int64> strides;
  };

  // The data parsed for the kDomain instruction.
  struct DomainData {
    std::unique_ptr<DomainMetadata> entry_metadata;
    std::unique_ptr<DomainMetadata> exit_metadata;
  };

  // Types of attributes.
  enum class AttrTy {
    kBool,
    kInt64,
    kInt32,
    kFloat,
    kString,
    kBracedInt64List,
    kBracedInt64ListList,
    kHloComputation,
    kBracedHloComputationList,
    kFftType,
    kComparisonDirection,
    kWindow,
    kConvolutionDimensionNumbers,
    kSharding,
    kFrontendAttributes,
    kParameterReplication,
    kInstructionList,
    kSliceRanges,
    kPaddingConfig,
    kMetadata,
    kFusionKind,
    kDistribution,
    kDomain,
    kPrecisionList,
    kShapeList,
    kEnum,
    kRandomAlgorithm,
    kAliasing,
  };

  struct AttrConfig {
    bool required;     // whether it's required or optional
    AttrTy attr_type;  // what type it is
    void* result;      // where to store the parsed result.
  };

  // attributes ::= (',' attribute)*
  //
  // Parses attributes given names and configs of the attributes. Each parsed
  // result is passed back through the result pointer in corresponding
  // AttrConfig. Note that the result pointer must point to a optional<T> typed
  // variable which outlives this function. Returns false on error. You should
  // not use the any of the results if this function failed.
  //
  // Example usage:
  //
  //  absl::flat_hash_map<std::string, AttrConfig> attrs;
  //  optional<int64> foo;
  //  attrs["foo"] = {/*required=*/false, AttrTy::kInt64, &foo};
  //  optional<Window> bar;
  //  attrs["bar"] = {/*required=*/true, AttrTy::kWindow, &bar};
  //  if (!ParseAttributes(attrs)) {
  //    return false; // Do not use 'foo' 'bar' if failed.
  //  }
  //  // Do something with 'bar'.
  //  if (foo) { // If attr foo is seen, do something with 'foo'. }
  //
  bool ParseAttributes(
      const absl::flat_hash_map<std::string, AttrConfig>& attrs);

  // sub_attributes ::= '{' (','? attribute)* '}'
  //
  // Usage is the same as ParseAttributes. See immediately above.
  bool ParseSubAttributes(
      const absl::flat_hash_map<std::string, AttrConfig>& attrs);

  // Parses one attribute. If it has already been seen, return error. Returns
  // true and adds to seen_attrs on success.
  //
  // Do not call this except in ParseAttributes or ParseSubAttributes.
  bool ParseAttributeHelper(
      const absl::flat_hash_map<std::string, AttrConfig>& attrs,
      absl::flat_hash_set<std::string>* seen_attrs);

  // Copy attributes from `attrs` to `message`, unless the attribute name is in
  // `non_proto_attrs`.
  bool CopyAttributeToProtoMessage(
      absl::flat_hash_set<std::string> non_proto_attrs,
      const absl::flat_hash_map<std::string, AttrConfig>& attrs,
      tensorflow::protobuf::Message* message);

  // Parses an attribute string into a protocol buffer `message`.
  // Since proto3 has no notion of mandatory fields, `required_attrs` gives the
  // set of mandatory attributes.
  // `non_proto_attrs` specifies attributes that are not written to the proto,
  // but added to the HloInstruction.
  bool ParseAttributesAsProtoMessage(
      const absl::flat_hash_map<std::string, AttrConfig>& non_proto_attrs,
      tensorflow::protobuf::Message* message);

  // Parses a name and finds the corresponding hlo computation.
  bool ParseComputationName(HloComputation** value);
  // Parses a list of names and finds the corresponding hlo instructions.
  bool ParseInstructionNames(std::vector<HloInstruction*>* instructions);
  // Pass expect_outer_curlies == true when parsing a Window in the context of a
  // larger computation.  Pass false when parsing a stand-alone Window string.
  bool ParseWindow(Window* window, bool expect_outer_curlies);
  bool ParseConvolutionDimensionNumbers(ConvolutionDimensionNumbers* dnums);
  bool ParsePaddingConfig(PaddingConfig* padding);
  bool ParseMetadata(OpMetadata* metadata);
  bool ParseSharding(OpSharding* sharding);
  bool ParseFrontendAttributes(FrontendAttributes* frontend_attributes);
  bool ParseSingleSharding(OpSharding* sharding, bool lbrace_pre_lexed);
  bool ParseParameterReplication(ParameterReplication* parameter_replication);
  bool ParseReplicaGroupsOnly(std::vector<ReplicaGroup>* replica_groups);

  // Parses the metadata behind a kDOmain instruction.
  bool ParseDomain(DomainData* domain);

  // Parses a sub-attribute of the window attribute, e.g.,size=1x2x3.
  bool ParseDxD(const std::string& name, std::vector<int64>* result);
  // Parses window's pad sub-attribute, e.g., pad=0_0x3x3.
  bool ParseWindowPad(std::vector<std::vector<int64>>* pad);

  bool ParseSliceRanges(SliceRanges* result);
  bool ParsePrecisionList(std::vector<PrecisionConfig::Precision>* result);
  bool ParseHloComputation(HloComputation** result);
  bool ParseHloComputationList(std::vector<HloComputation*>* result);
  bool ParseShapeList(std::vector<Shape>* result);
  bool ParseInt64List(const TokKind start, const TokKind end,
                      const TokKind delim, std::vector<int64>* result);
  bool ParseInt64ListList(const TokKind start, const TokKind end,
                          const TokKind delim,
                          std::vector<std::vector<int64>>* result);
  // 'parse_and_add_item' is an lambda to parse an element in the list and add
  // the parsed element to the result. It's supposed to capture the result.
  bool ParseList(const TokKind start, const TokKind end, const TokKind delim,
                 const std::function<bool()>& parse_and_add_item);

  bool ParseParamListToShape(Shape* shape, LocTy* shape_loc);
  bool ParseParamList();
  bool ParseName(std::string* result);
  bool ParseAttributeName(std::string* result);
  bool ParseString(std::string* result);
  bool ParseDimensionSizes(std::vector<int64>* dimension_sizes,
                           std::vector<bool>* dynamic_dimensions);
  bool ParseShape(Shape* result);
  bool ParseLayout(Layout* layout);
  bool ParseLayoutIntAttribute(int64* attr_value,
                               absl::string_view attr_description);
  bool ParseTiles(std::vector<Tile>* tiles);
  bool ParseOpcode(HloOpcode* result);
  bool ParseFftType(FftType* result);
  bool ParseComparisonDirection(ComparisonDirection* result);
  bool ParseFusionKind(HloInstruction::FusionKind* result);
  bool ParseRandomDistribution(RandomDistribution* result);
  bool ParseRandomAlgorithm(RandomAlgorithm* result);
  bool ParsePrecision(PrecisionConfig::Precision* result);
  bool ParseInt64(int64* result);
  bool ParseDouble(double* result);
  bool ParseComplex(std::complex<double>* result);
  bool ParseBool(bool* result);
  bool ParseToken(TokKind kind, const std::string& msg);

  using AliasingData =
      absl::flat_hash_map<ShapeIndex, HloInputOutputAliasConfig::Alias>;

  // Parses the aliasing information from string `s`, returns `false` if it
  // fails.
  bool ParseAliasing(AliasingData* data);

  bool ParseShapeIndex(ShapeIndex* out);

  // Returns true if the current token is the beginning of a shape.
  bool CanBeShape();
  // Returns true if the current token is the beginning of a
  // param_list_to_shape.
  bool CanBeParamListToShape();

  // Logs the current parsing line and the given message. Always returns false.
  bool TokenError(absl::string_view msg);
  bool Error(LocTy loc, absl::string_view msg);

  // If the current token is 'kind', eats it (i.e. lexes the next token) and
  // returns true.
  bool EatIfPresent(TokKind kind);

  // Adds the instruction to the pool. Returns false and emits an error if the
  // instruction already exists.
  bool AddInstruction(const std::string& name, HloInstruction* instruction,
                      LocTy name_loc);
  // Adds the computation to the pool. Returns false and emits an error if the
  // computation already exists.
  bool AddComputation(const std::string& name, HloComputation* computation,
                      LocTy name_loc);

  HloLexer lexer_;

  // A stack for the instruction names. The top of the stack stores the
  // instruction name table for the current scope.
  //
  // A instruction's name is unique among its scope (i.e. its parent
  // computation), but it's not necessarily unique among all computations in the
  // module. When there are multiple levels of nested computations, the same
  // name could appear in both an outer computation and an inner computation. So
  // we need a stack to make sure a name is only visible within its scope,
  std::vector<InstrNameTable> scoped_name_tables_;

  // A helper class which pushes and pops to an InstrNameTable stack via RAII.
  class Scope {
   public:
    explicit Scope(std::vector<InstrNameTable>* scoped_name_tables)
        : scoped_name_tables_(scoped_name_tables) {
      scoped_name_tables_->emplace_back();
    }
    ~Scope() { scoped_name_tables_->pop_back(); }

   private:
    std::vector<InstrNameTable>* scoped_name_tables_;
  };

  // Map from the computation name to the computation itself and its location.
  absl::flat_hash_map<std::string, std::pair<HloComputation*, LocTy>>
      computation_pool_;

  std::vector<std::unique_ptr<HloComputation>> computations_;
  std::vector<std::string> error_;

  // When an operand name cannot be resolved, this function is called to create
  // a parameter instruction with the given name and shape. It registers the
  // name, instruction, and a placeholder location in the name table. It returns
  // the newly-created instruction and the placeholder location. If `name` is
  // empty, this should create the parameter with a generated name. This is
  // supposed to be set and used only in ParseSingleInstruction.
  std::function<std::pair<HloInstruction*, LocTy>*(const std::string& name,
                                                   const Shape& shape)>
      create_missing_instruction_;
};

bool SplitToInt64s(absl::string_view s, char delim, std::vector<int64>* out) {
  for (const auto& split : absl::StrSplit(s, delim)) {
    int64 val;
    if (!absl::SimpleAtoi(split, &val)) {
      return false;
    }
    out->push_back(val);
  }
  return true;
}

// Creates replica groups from the provided nested array. groups[i] represents
// the replica ids for group 'i'.
std::vector<ReplicaGroup> CreateReplicaGroups(
    absl::Span<const std::vector<int64>> groups) {
  std::vector<ReplicaGroup> replica_groups;
  absl::c_transform(groups, std::back_inserter(replica_groups),
                    [](const std::vector<int64>& ids) {
                      ReplicaGroup group;
                      *group.mutable_replica_ids() = {ids.begin(), ids.end()};
                      return group;
                    });
  return replica_groups;
}

bool HloParserImpl::Error(LocTy loc, absl::string_view msg) {
  auto line_col = lexer_.GetLineAndColumn(loc);
  const unsigned line = line_col.first;
  const unsigned col = line_col.second;
  std::vector<std::string> error_lines;
  error_lines.push_back(
      StrCat("was parsing ", line, ":", col, ": error: ", msg));
  error_lines.emplace_back(lexer_.GetLine(loc));
  error_lines.push_back(col == 0 ? "" : StrCat(std::string(col - 1, ' '), "^"));

  error_.push_back(StrJoin(error_lines, "\n"));
  VLOG(1) << "Error: " << error_.back();
  return false;
}

bool HloParserImpl::TokenError(absl::string_view msg) {
  return Error(lexer_.GetLoc(), msg);
}

Status HloParserImpl::Run(HloModule* module) {
  lexer_.Lex();
  if (lexer_.GetKind() == TokKind::kw_HloModule) {
    // This means that the text contains a full HLO module.
    if (!ParseHloModule(module)) {
      return InvalidArgument(
          "Syntax error when trying to parse the text as a HloModule:\n%s",
          GetError());
    }
    return Status::OK();
  }
  // This means that the text is a single HLO instruction.
  if (!ParseSingleInstruction(module)) {
    return InvalidArgument(
        "Syntax error when trying to parse the text as a single "
        "HloInstruction:\n%s",
        GetError());
  }
  return Status::OK();
}

std::pair<HloInstruction*, HloParserImpl::LocTy>*
HloParserImpl::FindInstruction(const std::string& name,
                               const optional<Shape>& shape) {
  std::pair<HloInstruction*, LocTy>* instr = nullptr;
  if (!name.empty()) {
    instr = tensorflow::gtl::FindOrNull(current_name_table(), name);
  }

  // Potentially call the missing instruction hook.
  if (instr == nullptr && create_missing_instruction_ != nullptr &&
      scoped_name_tables_.size() == 1) {
    if (!shape.has_value()) {
      Error(lexer_.GetLoc(),
            "Operand had no shape in HLO text; cannot create parameter for "
            "single-instruction module.");
      return nullptr;
    }
    return create_missing_instruction_(name, *shape);
  }

  if (instr != nullptr && shape.has_value() &&
      !ShapeUtil::Compatible(instr->first->shape(), shape.value())) {
    Error(
        lexer_.GetLoc(),
        StrCat("The declared operand shape ",
               ShapeUtil::HumanStringWithLayout(shape.value()),
               " is not compatible with the shape of the operand instruction ",
               ShapeUtil::HumanStringWithLayout(instr->first->shape()), "."));
    return nullptr;
  }

  return instr;
}

bool HloParserImpl::ParseShapeIndex(ShapeIndex* out) {
  if (!ParseToken(TokKind::kLbrace, "Expects '{' at the start of ShapeIndex")) {
    return false;
  }

  std::vector<int64> idxs;
  while (lexer_.GetKind() != TokKind::kRbrace) {
    int64 idx;
    if (!ParseInt64(&idx)) {
      return false;
    }
    idxs.push_back(idx);
    if (!EatIfPresent(TokKind::kComma)) {
      break;
    }
  }
  if (!ParseToken(TokKind::kRbrace, "Expects '}' at the end of ShapeIndex")) {
    return false;
  }
  *out = ShapeIndex(idxs.begin(), idxs.end());
  return true;
}

bool HloParserImpl::ParseAliasing(AliasingData* data) {
  if (!ParseToken(TokKind::kLbrace,
                  "Expects '{' at the start of aliasing description")) {
    return false;
  }

  while (lexer_.GetKind() != TokKind::kRbrace) {
    ShapeIndex out;
    if (!ParseShapeIndex(&out)) {
      return false;
    }
    std::string errmsg =
        "Expected format: <output_shape_index>: (<input_param>, "
        "<input_param_shape_index>)";
    if (!ParseToken(TokKind::kColon, errmsg)) {
      return false;
    }
    if (!ParseToken(TokKind::kLparen, errmsg)) {
      return false;
    }
    int64 param_num;
    ParseInt64(&param_num);
    if (!ParseToken(TokKind::kComma, errmsg)) {
      return false;
    }
    ShapeIndex param_idx;
    if (!ParseShapeIndex(&param_idx)) {
      return false;
    }
    HloInputOutputAliasConfig::AliasKind alias_kind =
        HloInputOutputAliasConfig::kUserAlias;
    if (EatIfPresent(TokKind::kComma)) {
      std::string type;
      ParseName(&type);
      if (type == "SYSTEM") {
        alias_kind = HloInputOutputAliasConfig::kSystemAlias;
      } else if (type == "USER") {
        alias_kind = HloInputOutputAliasConfig::kUserAlias;
      } else {
        return TokenError("Unexpected aliasing kind; expected SYSTEM or USER");
      }
    }
    data->emplace(std::piecewise_construct, std::forward_as_tuple(out),
                  std::forward_as_tuple(alias_kind, param_num, param_idx));
    if (!ParseToken(TokKind::kRparen, errmsg)) {
      return false;
    }

    if (!EatIfPresent(TokKind::kComma)) {
      break;
    }
  }
  if (!ParseToken(TokKind::kRbrace,
                  "Expects '}' at the end of aliasing description")) {
    return false;
  }
  return true;
}

// ::= 'HloModule' name computations
bool HloParserImpl::ParseHloModule(HloModule* module) {
  if (lexer_.GetKind() != TokKind::kw_HloModule) {
    return TokenError("expects HloModule");
  }
  // Eat 'HloModule'
  lexer_.Lex();

  std::string name;
  if (!ParseName(&name)) {
    return false;
  }

  absl::optional<bool> is_scheduled;
  absl::optional<AliasingData> aliasing_data;
  absl::flat_hash_map<std::string, AttrConfig> attrs;

  attrs["is_scheduled"] = {/*required=*/false, AttrTy::kBool, &is_scheduled};
  attrs["input_output_alias"] = {/*required=*/false, AttrTy::kAliasing,
                                 &aliasing_data};
  if (!ParseAttributes(attrs)) {
    return false;
  }
  module->set_name(name);
  if (!ParseComputations(module)) {
    return false;
  }

  if (is_scheduled.has_value() && *is_scheduled) {
    TF_CHECK_OK(module->set_schedule(ScheduleFromInstructionOrder(module)));
  }
  if (aliasing_data) {
    HloInputOutputAliasConfig alias_config(module->result_shape());
    for (auto& p : *aliasing_data) {
      Status st =
          alias_config.SetUpAlias(p.first, p.second.parameter_number,
                                  p.second.parameter_index, p.second.kind);
      if (!st.ok()) {
        return TokenError(st.error_message());
      }
    }
    module->input_output_alias_config() = alias_config;
  }

  return true;
}

// computations ::= (computation)+
bool HloParserImpl::ParseComputations(HloModule* module) {
  HloComputation* entry_computation = nullptr;
  do {
    if (!ParseComputation(&entry_computation)) {
      return false;
    }
  } while (lexer_.GetKind() != TokKind::kEof);

  for (int i = 0; i < computations_.size(); i++) {
    // If entry_computation is not nullptr, it means the computation it pointed
    // to is marked with "ENTRY"; otherwise, no computation is marked with
    // "ENTRY", and we use the last computation as the entry computation. We
    // add the non-entry computations as embedded computations to the module.
    if ((entry_computation != nullptr &&
         computations_[i].get() != entry_computation) ||
        (entry_computation == nullptr && i != computations_.size() - 1)) {
      module->AddEmbeddedComputation(std::move(computations_[i]));
      continue;
    }
    auto computation = module->AddEntryComputation(std::move(computations_[i]));
    // The parameters and result layouts were set to default layout. Here we
    // set the layouts to what the hlo text says.
    for (int p = 0; p < computation->num_parameters(); p++) {
      const Shape& param_shape = computation->parameter_instruction(p)->shape();
      TF_CHECK_OK(module->mutable_entry_computation_layout()
                      ->mutable_parameter_layout(p)
                      ->CopyLayoutFromShape(param_shape));
    }
    const Shape& result_shape = computation->root_instruction()->shape();
    TF_CHECK_OK(module->mutable_entry_computation_layout()
                    ->mutable_result_layout()
                    ->CopyLayoutFromShape(result_shape));
  }
  return true;
}

// computation ::= ('ENTRY')? name (param_list_to_shape)? instruction_list
bool HloParserImpl::ParseComputation(HloComputation** entry_computation) {
  LocTy maybe_entry_loc = lexer_.GetLoc();
  const bool is_entry_computation = EatIfPresent(TokKind::kw_ENTRY);

  std::string name;
  LocTy name_loc = lexer_.GetLoc();
  if (!ParseName(&name)) {
    return false;
  }

  LocTy shape_loc = nullptr;
  Shape shape;
  if (CanBeParamListToShape() && !ParseParamListToShape(&shape, &shape_loc)) {
    return false;
  }

  HloComputation* computation = nullptr;
  if (!ParseInstructionList(&computation, name)) {
    return false;
  }

  // If param_list_to_shape was present, check compatibility.
  if (shape_loc != nullptr &&
      !ShapeUtil::Compatible(computation->root_instruction()->shape(), shape)) {
    return Error(
        shape_loc,
        StrCat(
            "Shape of computation ", name, ", ", ShapeUtil::HumanString(shape),
            ", is not compatible with that of its root instruction ",
            computation->root_instruction()->name(), ", ",
            ShapeUtil::HumanString(computation->root_instruction()->shape())));
  }

  if (is_entry_computation) {
    if (*entry_computation != nullptr) {
      return Error(maybe_entry_loc, "expects only one ENTRY");
    }
    *entry_computation = computation;
  }

  return AddComputation(name, computation, name_loc);
}

// instruction_list ::= '{' instruction_list1 '}'
// instruction_list1 ::= (instruction)+
bool HloParserImpl::ParseInstructionList(HloComputation** computation,
                                         const std::string& computation_name) {
  Scope scope(&scoped_name_tables_);
  HloComputation::Builder builder(computation_name);
  if (!ParseToken(TokKind::kLbrace,
                  "expects '{' at the beginning of instruction list.")) {
    return false;
  }
  std::string root_name;
  do {
    if (!ParseInstruction(&builder, &root_name)) {
      return false;
    }
  } while (lexer_.GetKind() != TokKind::kRbrace);
  if (!ParseToken(TokKind::kRbrace,
                  "expects '}' at the end of instruction list.")) {
    return false;
  }
  HloInstruction* root = nullptr;
  if (!root_name.empty()) {
    std::pair<HloInstruction*, LocTy>* root_node =
        tensorflow::gtl::FindOrNull(current_name_table(), root_name);

    // This means some instruction was marked as ROOT but we didn't find it in
    // the pool, which should not happen.
    if (root_node == nullptr) {
      // LOG(FATAL) crashes the program by calling abort().
      LOG(FATAL) << "instruction " << root_name
                 << " was marked as ROOT but the parser has not seen it before";
    }
    root = root_node->first;
  }

  // Now root can be either an existing instruction or a nullptr. If it's a
  // nullptr, the implementation of Builder will set the last instruction as
  // the root instruction.
  computations_.emplace_back(builder.Build(root));
  *computation = computations_.back().get();
  return true;
}

// instruction ::= ('ROOT')? name '=' shape opcode operands (attribute)*
bool HloParserImpl::ParseInstruction(HloComputation::Builder* builder,
                                     std::string* root_name) {
  std::string name;
  LocTy maybe_root_loc = lexer_.GetLoc();
  bool is_root = EatIfPresent(TokKind::kw_ROOT);

  const LocTy name_loc = lexer_.GetLoc();
  if (!ParseName(&name) ||
      !ParseToken(TokKind::kEqual, "expects '=' in instruction")) {
    return false;
  }

  if (is_root) {
    if (!root_name->empty()) {
      return Error(maybe_root_loc, "one computation should have only one ROOT");
    }
    *root_name = name;
  }

  return ParseInstructionRhs(builder, name, name_loc);
}

bool HloParserImpl::ParseInstructionRhs(HloComputation::Builder* builder,
                                        const std::string& name,
                                        LocTy name_loc) {
  Shape shape;
  HloOpcode opcode;
  std::vector<HloInstruction*> operands;

  if (!ParseShape(&shape) || !ParseOpcode(&opcode)) {
    return false;
  }

  // Add optional attributes. These are added to any HloInstruction type if
  // present.
  absl::flat_hash_map<std::string, AttrConfig> attrs;
  optional<OpSharding> sharding;
  optional<FrontendAttributes> frontend_attributes;
  attrs["sharding"] = {/*required=*/false, AttrTy::kSharding, &sharding};
  attrs["frontend_attributes"] = {
      /*required=*/false, AttrTy::kFrontendAttributes, &frontend_attributes};
  optional<ParameterReplication> parameter_replication;
  attrs["parameter_replication"] = {/*required=*/false,
                                    AttrTy::kParameterReplication,
                                    &parameter_replication};
  optional<std::vector<HloInstruction*>> predecessors;
  attrs["control-predecessors"] = {/*required=*/false, AttrTy::kInstructionList,
                                   &predecessors};
  optional<OpMetadata> metadata;
  attrs["metadata"] = {/*required=*/false, AttrTy::kMetadata, &metadata};

  optional<std::string> backend_config;
  attrs["backend_config"] = {/*required=*/false, AttrTy::kString,
                             &backend_config};
  optional<std::vector<int64>> outer_dimension_partitions;
  attrs["outer_dimension_partitions"] = {/*required=*/false,
                                         AttrTy::kBracedInt64List,
                                         &outer_dimension_partitions};

  HloInstruction* instruction;
  switch (opcode) {
    case HloOpcode::kParameter: {
      int64 parameter_number;
      if (!ParseToken(TokKind::kLparen,
                      "expects '(' before parameter number") ||
          !ParseInt64(&parameter_number)) {
        return false;
      }
      if (parameter_number < 0) {
        Error(lexer_.GetLoc(), "parameter number must be >= 0");
        return false;
      }
      if (!ParseToken(TokKind::kRparen, "expects ')' after parameter number") ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateParameter(parameter_number, shape, name));
      break;
    }
    case HloOpcode::kConstant: {
      Literal literal;
      if (!ParseToken(TokKind::kLparen,
                      "expects '(' before constant literal") ||
          !ParseLiteral(&literal, shape) ||
          !ParseToken(TokKind::kRparen, "expects ')' after constant literal") ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateConstant(std::move(literal)));
      break;
    }
    case HloOpcode::kIota: {
      optional<int64> iota_dimension;
      attrs["iota_dimension"] = {/*required=*/true, AttrTy::kInt64,
                                 &iota_dimension};
      if (!ParseOperands(&operands, /*expected_size=*/0) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateIota(shape, *iota_dimension));
      break;
    }
    // Unary ops.
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyStart:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kTanh: {
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateUnary(shape, opcode, operands[0]));
      break;
    }
    // Binary ops.
    case HloOpcode::kAdd:
    case HloOpcode::kDivide:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
    case HloOpcode::kAtan2:
    case HloOpcode::kComplex:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical: {
      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateBinary(
          shape, opcode, operands[0], operands[1]));
      break;
    }
    // Ternary ops.
    case HloOpcode::kClamp:
    case HloOpcode::kSelect:
    case HloOpcode::kTupleSelect: {
      if (!ParseOperands(&operands, /*expected_size=*/3) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateTernary(
          shape, opcode, operands[0], operands[1], operands[2]));
      break;
    }
    // Other supported ops.
    case HloOpcode::kConvert: {
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateConvert(shape, operands[0]));
      break;
    }
    case HloOpcode::kBitcastConvert: {
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateBitcastConvert(shape, operands[0]));
      break;
    }
    case HloOpcode::kAllGather: {
      optional<std::vector<std::vector<int64>>> tmp_groups;
      optional<std::vector<int64>> replica_group_ids;
      optional<int64> channel_id;
      optional<std::vector<int64>> dimensions;
      optional<bool> constrain_layout;
      optional<bool> use_global_device_ids;
      attrs["replica_groups"] = {/*required=*/false,
                                 AttrTy::kBracedInt64ListList, &tmp_groups};
      attrs["channel_id"] = {/*required=*/false, AttrTy::kInt64, &channel_id};
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions};
      attrs["constrain_layout"] = {/*required=*/false, AttrTy::kBool,
                                   &constrain_layout};
      attrs["use_global_device_ids"] = {/*required=*/false, AttrTy::kBool,
                                        &use_global_device_ids};
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      std::vector<ReplicaGroup> replica_groups;
      if (tmp_groups) {
        replica_groups = CreateReplicaGroups(*tmp_groups);
      }
      instruction = builder->AddInstruction(HloInstruction::CreateAllGather(
          shape, operands[0], dimensions->at(0), replica_groups,
          constrain_layout ? *constrain_layout : false, channel_id,
          use_global_device_ids ? *use_global_device_ids : false));
      break;
    }
    case HloOpcode::kAllReduce: {
      optional<std::vector<std::vector<int64>>> tmp_groups;
      optional<HloComputation*> to_apply;
      optional<std::vector<int64>> replica_group_ids;
      optional<int64> channel_id;
      optional<bool> constrain_layout;
      optional<bool> use_global_device_ids;
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &to_apply};
      attrs["replica_groups"] = {/*required=*/false,
                                 AttrTy::kBracedInt64ListList, &tmp_groups};
      attrs["channel_id"] = {/*required=*/false, AttrTy::kInt64, &channel_id};
      attrs["constrain_layout"] = {/*required=*/false, AttrTy::kBool,
                                   &constrain_layout};
      attrs["use_global_device_ids"] = {/*required=*/false, AttrTy::kBool,
                                        &use_global_device_ids};
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      std::vector<ReplicaGroup> replica_groups;
      if (tmp_groups) {
        replica_groups = CreateReplicaGroups(*tmp_groups);
      }
      instruction = builder->AddInstruction(HloInstruction::CreateAllReduce(
          shape, operands, *to_apply, replica_groups,
          constrain_layout ? *constrain_layout : false, channel_id,
          use_global_device_ids ? *use_global_device_ids : false));
      break;
    }
    case HloOpcode::kAllToAll: {
      optional<std::vector<std::vector<int64>>> tmp_groups;
      attrs["replica_groups"] = {/*required=*/false,
                                 AttrTy::kBracedInt64ListList, &tmp_groups};
      optional<int64> channel_id;
      attrs["channel_id"] = {/*required=*/false, AttrTy::kInt64, &channel_id};
      optional<std::vector<int64>> dimensions;
      attrs["dimensions"] = {/*required=*/false, AttrTy::kBracedInt64List,
                             &dimensions};
      optional<bool> constrain_layout;
      attrs["constrain_layout"] = {/*required=*/false, AttrTy::kBool,
                                   &constrain_layout};
      if (!ParseOperands(&operands) || !ParseAttributes(attrs) ||
          (dimensions && dimensions->size() != 1)) {
        return false;
      }
      std::vector<ReplicaGroup> replica_groups;
      if (tmp_groups) {
        replica_groups = CreateReplicaGroups(*tmp_groups);
      }
      optional<int64> split_dimension;
      if (dimensions) {
        split_dimension = dimensions->at(0);
      }
      instruction = builder->AddInstruction(HloInstruction::CreateAllToAll(
          shape, operands, replica_groups,
          constrain_layout ? *constrain_layout : false, channel_id,
          split_dimension));
      break;
    }
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart: {
      optional<std::vector<std::vector<int64>>> source_targets;
      attrs["source_target_pairs"] = {
          /*required=*/true, AttrTy::kBracedInt64ListList, &source_targets};
      optional<int64> channel_id;
      attrs["channel_id"] = {/*required=*/false, AttrTy::kInt64, &channel_id};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      std::vector<std::pair<int64, int64>> pairs(source_targets->size());
      for (int i = 0; i < pairs.size(); i++) {
        if ((*source_targets)[i].size() != 2) {
          return TokenError(
              "expects 'source_target_pairs=' to be a list of pairs");
        }
        pairs[i].first = (*source_targets)[i][0];
        pairs[i].second = (*source_targets)[i][1];
      }
      if (opcode == HloOpcode::kCollectivePermute) {
        instruction =
            builder->AddInstruction(HloInstruction::CreateCollectivePermute(
                shape, operands[0], pairs, channel_id));
      } else if (opcode == HloOpcode::kCollectivePermuteStart) {
        instruction = builder->AddInstruction(
            HloInstruction::CreateCollectivePermuteStart(shape, operands[0],
                                                         pairs, channel_id));
      } else {
        LOG(FATAL) << "Expect opcode to be CollectivePermute or "
                      "CollectivePermuteStart, but got "
                   << HloOpcodeString(opcode);
      }
      break;
    }
    case HloOpcode::kReplicaId: {
      if (!ParseOperands(&operands, /*expected_size=*/0) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateReplicaId());
      break;
    }
    case HloOpcode::kPartitionId: {
      if (!ParseOperands(&operands, /*expected_size=*/0) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction =
          builder->AddInstruction(HloInstruction::CreatePartitionId());
      break;
    }
    case HloOpcode::kReshape: {
      optional<int64> inferred_dimension;
      attrs["inferred_dimension"] = {/*required=*/false, AttrTy::kInt64,
                                     &inferred_dimension};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateReshape(
          shape, operands[0], inferred_dimension.value_or(-1)));
      break;
    }
    case HloOpcode::kAfterAll: {
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      if (operands.empty()) {
        instruction = builder->AddInstruction(HloInstruction::CreateToken());
      } else {
        instruction =
            builder->AddInstruction(HloInstruction::CreateAfterAll(operands));
      }
      break;
    }
    case HloOpcode::kAddDependency: {
      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateAddDependency(operands[0], operands[1]));
      break;
    }
    case HloOpcode::kSort: {
      optional<std::vector<int64>> dimensions;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions};
      optional<bool> is_stable = false;
      attrs["is_stable"] = {/*required=*/false, AttrTy::kBool, &is_stable};
      optional<HloComputation*> to_apply;
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &to_apply};
      if (!ParseOperands(&operands) || !ParseAttributes(attrs) ||
          dimensions->size() != 1) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateSort(shape, dimensions->at(0), operands,
                                     to_apply.value(), is_stable.value()));
      break;
    }
    case HloOpcode::kTuple: {
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      instruction =
          builder->AddInstruction(HloInstruction::CreateTuple(operands));
      break;
    }
    case HloOpcode::kWhile: {
      optional<HloComputation*> condition;
      optional<HloComputation*> body;
      attrs["condition"] = {/*required=*/true, AttrTy::kHloComputation,
                            &condition};
      attrs["body"] = {/*required=*/true, AttrTy::kHloComputation, &body};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateWhile(
          shape, *condition, *body, /*init=*/operands[0]));
      break;
    }
    case HloOpcode::kRecv: {
      optional<int64> channel_id;
      // If the is_host_transfer attribute is not present then default to false.
      optional<bool> is_host_transfer = false;
      attrs["channel_id"] = {/*required=*/true, AttrTy::kInt64, &channel_id};
      attrs["is_host_transfer"] = {/*required=*/false, AttrTy::kBool,
                                   &is_host_transfer};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      // If the is_host_transfer attribute is not present then default to false.
      instruction = builder->AddInstruction(HloInstruction::CreateRecv(
          shape.tuple_shapes(0), operands[0], *channel_id, *is_host_transfer));
      break;
    }
    case HloOpcode::kRecvDone: {
      optional<int64> channel_id;
      // If the is_host_transfer attribute is not present then default to false.
      optional<bool> is_host_transfer = false;
      attrs["channel_id"] = {/*required=*/true, AttrTy::kInt64, &channel_id};
      attrs["is_host_transfer"] = {/*required=*/false, AttrTy::kBool,
                                   &is_host_transfer};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      if (dynamic_cast<const HloChannelInstruction*>(operands[0]) == nullptr) {
        return false;
      }
      if (channel_id != operands[0]->channel_id()) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateRecvDone(operands[0], *is_host_transfer));
      break;
    }
    case HloOpcode::kSend: {
      optional<int64> channel_id;
      // If the is_host_transfer attribute is not present then default to false.
      optional<bool> is_host_transfer = false;
      attrs["channel_id"] = {/*required=*/true, AttrTy::kInt64, &channel_id};
      attrs["is_host_transfer"] = {/*required=*/false, AttrTy::kBool,
                                   &is_host_transfer};
      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateSend(
          operands[0], operands[1], *channel_id, *is_host_transfer));
      break;
    }
    case HloOpcode::kSendDone: {
      optional<int64> channel_id;
      // If the is_host_transfer attribute is not present then default to false.
      optional<bool> is_host_transfer = false;
      attrs["channel_id"] = {/*required=*/true, AttrTy::kInt64, &channel_id};
      attrs["is_host_transfer"] = {/*required=*/false, AttrTy::kBool,
                                   &is_host_transfer};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      if (dynamic_cast<const HloChannelInstruction*>(operands[0]) == nullptr) {
        return false;
      }
      if (channel_id != operands[0]->channel_id()) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateSendDone(operands[0], *is_host_transfer));
      break;
    }
    case HloOpcode::kGetTupleElement: {
      optional<int64> index;
      attrs["index"] = {/*required=*/true, AttrTy::kInt64, &index};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateGetTupleElement(shape, operands[0], *index));
      break;
    }
    case HloOpcode::kCall: {
      optional<HloComputation*> to_apply;
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &to_apply};
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateCall(shape, operands, *to_apply));
      break;
    }
    case HloOpcode::kReduceWindow: {
      optional<HloComputation*> reduce_computation;
      optional<Window> window;
      attrs["window"] = {/*required=*/false, AttrTy::kWindow, &window};
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &reduce_computation};
      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      if (!window) {
        window.emplace();
      }
      instruction = builder->AddInstruction(HloInstruction::CreateReduceWindow(
          shape, /*operand=*/operands[0], /*init_value=*/operands[1], *window,
          *reduce_computation));
      break;
    }
    case HloOpcode::kConvolution: {
      optional<Window> window;
      optional<ConvolutionDimensionNumbers> dnums;
      optional<int64> feature_group_count;
      optional<int64> batch_group_count;
      attrs["window"] = {/*required=*/false, AttrTy::kWindow, &window};
      attrs["dim_labels"] = {/*required=*/true,
                             AttrTy::kConvolutionDimensionNumbers, &dnums};
      attrs["feature_group_count"] = {/*required=*/false, AttrTy::kInt64,
                                      &feature_group_count};
      attrs["batch_group_count"] = {/*required=*/false, AttrTy::kInt64,
                                    &batch_group_count};
      optional<std::vector<PrecisionConfig::Precision>> operand_precision;
      attrs["operand_precision"] = {/*required=*/false, AttrTy::kPrecisionList,
                                    &operand_precision};
      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      if (!window) {
        window.emplace();
      }
      if (!feature_group_count) {
        feature_group_count = 1;
      }
      if (!batch_group_count) {
        batch_group_count = 1;
      }
      PrecisionConfig precision_config;
      if (operand_precision) {
        *precision_config.mutable_operand_precision() = {
            operand_precision->begin(), operand_precision->end()};
      } else {
        precision_config.mutable_operand_precision()->Resize(
            operands.size(), PrecisionConfig::DEFAULT);
      }
      instruction = builder->AddInstruction(HloInstruction::CreateConvolve(
          shape, /*lhs=*/operands[0], /*rhs=*/operands[1],
          feature_group_count.value(), batch_group_count.value(), *window,
          *dnums, precision_config));
      break;
    }
    case HloOpcode::kFft: {
      optional<FftType> fft_type;
      optional<std::vector<int64>> fft_length;
      attrs["fft_type"] = {/*required=*/true, AttrTy::kFftType, &fft_type};
      attrs["fft_length"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &fft_length};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateFft(
          shape, operands[0], *fft_type, *fft_length));
      break;
    }
    case HloOpcode::kTriangularSolve: {
      TriangularSolveOptions options;
      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributesAsProtoMessage(
              /*non_proto_attrs=*/attrs, &options)) {
        return false;
      }
      instruction =
          builder->AddInstruction(HloInstruction::CreateTriangularSolve(
              shape, operands[0], operands[1], options));
      break;
    }
    case HloOpcode::kCompare: {
      optional<ComparisonDirection> direction;
      attrs["direction"] = {/*required=*/true, AttrTy::kComparisonDirection,
                            &direction};
      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateCompare(
          shape, operands[0], operands[1], *direction));
      break;
    }
    case HloOpcode::kCholesky: {
      CholeskyOptions options;
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributesAsProtoMessage(
              /*non_proto_attrs=*/attrs, &options)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateCholesky(shape, operands[0], options));
      break;
    }
    case HloOpcode::kBroadcast: {
      optional<std::vector<int64>> broadcast_dimensions;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &broadcast_dimensions};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateBroadcast(
          shape, operands[0], *broadcast_dimensions));
      break;
    }
    case HloOpcode::kConcatenate: {
      optional<std::vector<int64>> dimensions;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions};
      if (!ParseOperands(&operands) || !ParseAttributes(attrs) ||
          dimensions->size() != 1) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateConcatenate(
          shape, operands, dimensions->at(0)));
      break;
    }
    case HloOpcode::kMap: {
      optional<HloComputation*> to_apply;
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &to_apply};
      optional<std::vector<int64>> dimensions;
      attrs["dimensions"] = {/*required=*/false, AttrTy::kBracedInt64List,
                             &dimensions};
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateMap(shape, operands, *to_apply));
      break;
    }
    case HloOpcode::kReduce: {
      auto loc = lexer_.GetLoc();

      optional<HloComputation*> reduce_computation;
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &reduce_computation};
      optional<std::vector<int64>> dimensions_to_reduce;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions_to_reduce};
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      if (operands.size() % 2) {
        return Error(loc, StrCat("expects an even number of operands, but has ",
                                 operands.size(), " operands"));
      }
      instruction = builder->AddInstruction(HloInstruction::CreateReduce(
          shape, /*operands=*/
          absl::Span<HloInstruction* const>(operands).subspan(
              0, operands.size() / 2),
          /*init_values=*/
          absl::Span<HloInstruction* const>(operands).subspan(operands.size() /
                                                              2),
          *dimensions_to_reduce, *reduce_computation));
      break;
    }
    case HloOpcode::kReverse: {
      optional<std::vector<int64>> dimensions;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateReverse(shape, operands[0], *dimensions));
      break;
    }
    case HloOpcode::kSelectAndScatter: {
      optional<HloComputation*> select;
      attrs["select"] = {/*required=*/true, AttrTy::kHloComputation, &select};
      optional<HloComputation*> scatter;
      attrs["scatter"] = {/*required=*/true, AttrTy::kHloComputation, &scatter};
      optional<Window> window;
      attrs["window"] = {/*required=*/false, AttrTy::kWindow, &window};
      if (!ParseOperands(&operands, /*expected_size=*/3) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      if (!window) {
        window.emplace();
      }
      instruction =
          builder->AddInstruction(HloInstruction::CreateSelectAndScatter(
              shape, /*operand=*/operands[0], *select, *window,
              /*source=*/operands[1], /*init_value=*/operands[2], *scatter));
      break;
    }
    case HloOpcode::kSlice: {
      optional<SliceRanges> slice_ranges;
      attrs["slice"] = {/*required=*/true, AttrTy::kSliceRanges, &slice_ranges};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateSlice(
          shape, operands[0], slice_ranges->starts, slice_ranges->limits,
          slice_ranges->strides));
      break;
    }
    case HloOpcode::kDynamicSlice: {
      optional<std::vector<int64>> dynamic_slice_sizes;
      attrs["dynamic_slice_sizes"] = {
          /*required=*/true, AttrTy::kBracedInt64List, &dynamic_slice_sizes};
      LocTy loc = lexer_.GetLoc();
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      if (operands.empty()) {
        return Error(loc, "Expected at least one operand.");
      }
      if (!(operands.size() == 2 && operands[1]->shape().rank() == 1) &&
          operands.size() != 1 + operands[0]->shape().rank()) {
        return Error(loc, "Wrong number of operands.");
      }
      instruction = builder->AddInstruction(HloInstruction::CreateDynamicSlice(
          shape, /*operand=*/operands[0],
          /*start_indices=*/absl::MakeSpan(operands).subspan(1),
          *dynamic_slice_sizes));
      break;
    }
    case HloOpcode::kDynamicUpdateSlice: {
      LocTy loc = lexer_.GetLoc();
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      if (operands.size() < 2) {
        return Error(loc, "Expected at least two operands.");
      }
      if (!(operands.size() == 3 && operands[2]->shape().rank() == 1) &&
          operands.size() != 2 + operands[0]->shape().rank()) {
        return Error(loc, "Wrong number of operands.");
      }
      instruction =
          builder->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
              shape, /*operand=*/operands[0], /*update=*/operands[1],
              /*start_indices=*/absl::MakeSpan(operands).subspan(2)));
      break;
    }
    case HloOpcode::kTranspose: {
      optional<std::vector<int64>> dimensions;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateTranspose(shape, operands[0], *dimensions));
      break;
    }
    case HloOpcode::kBatchNormTraining: {
      optional<float> epsilon;
      attrs["epsilon"] = {/*required=*/true, AttrTy::kFloat, &epsilon};
      optional<int64> feature_index;
      attrs["feature_index"] = {/*required=*/true, AttrTy::kInt64,
                                &feature_index};
      if (!ParseOperands(&operands, /*expected_size=*/3) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction =
          builder->AddInstruction(HloInstruction::CreateBatchNormTraining(
              shape, /*operand=*/operands[0], /*scale=*/operands[1],
              /*offset=*/operands[2], *epsilon, *feature_index));
      break;
    }
    case HloOpcode::kBatchNormInference: {
      optional<float> epsilon;
      attrs["epsilon"] = {/*required=*/true, AttrTy::kFloat, &epsilon};
      optional<int64> feature_index;
      attrs["feature_index"] = {/*required=*/true, AttrTy::kInt64,
                                &feature_index};
      if (!ParseOperands(&operands, /*expected_size=*/5) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction =
          builder->AddInstruction(HloInstruction::CreateBatchNormInference(
              shape, /*operand=*/operands[0], /*scale=*/operands[1],
              /*offset=*/operands[2], /*mean=*/operands[3],
              /*variance=*/operands[4], *epsilon, *feature_index));
      break;
    }
    case HloOpcode::kBatchNormGrad: {
      optional<float> epsilon;
      attrs["epsilon"] = {/*required=*/true, AttrTy::kFloat, &epsilon};
      optional<int64> feature_index;
      attrs["feature_index"] = {/*required=*/true, AttrTy::kInt64,
                                &feature_index};
      if (!ParseOperands(&operands, /*expected_size=*/5) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateBatchNormGrad(
          shape, /*operand=*/operands[0], /*scale=*/operands[1],
          /*mean=*/operands[2], /*variance=*/operands[3],
          /*grad_output=*/operands[4], *epsilon, *feature_index));
      break;
    }
    case HloOpcode::kPad: {
      optional<PaddingConfig> padding;
      attrs["padding"] = {/*required=*/true, AttrTy::kPaddingConfig, &padding};
      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreatePad(
          shape, operands[0], /*padding_value=*/operands[1], *padding));
      break;
    }
    case HloOpcode::kFusion: {
      optional<HloComputation*> fusion_computation;
      attrs["calls"] = {/*required=*/true, AttrTy::kHloComputation,
                        &fusion_computation};
      optional<HloInstruction::FusionKind> fusion_kind;
      attrs["kind"] = {/*required=*/true, AttrTy::kFusionKind, &fusion_kind};
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateFusion(
          shape, *fusion_kind, operands, *fusion_computation));
      break;
    }
    case HloOpcode::kInfeed: {
      optional<std::string> config;
      attrs["infeed_config"] = {/*required=*/false, AttrTy::kString, &config};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      // We need to know the infeed data shape to construct the infeed
      // instruction. This is the zero-th element of the tuple-shaped output of
      // the infeed instruction. ShapeUtil::GetTupleElementShape will check fail
      // if the shape is not a non-empty tuple, so add guard so an error message
      // can be emitted instead of a check fail
      if (!shape.IsTuple() && !ShapeUtil::IsEmptyTuple(shape)) {
        return Error(lexer_.GetLoc(),
                     "infeed must have a non-empty tuple shape");
      }
      instruction = builder->AddInstruction(HloInstruction::CreateInfeed(
          ShapeUtil::GetTupleElementShape(shape, 0), operands[0],
          config ? *config : ""));
      break;
    }
    case HloOpcode::kOutfeed: {
      optional<std::string> config;
      attrs["outfeed_config"] = {/*required=*/false, AttrTy::kString, &config};
      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateOutfeed(operands[0]->shape(), operands[0],
                                        operands[1], config ? *config : ""));
      break;
    }
    case HloOpcode::kRng: {
      optional<RandomDistribution> distribution;
      attrs["distribution"] = {/*required=*/true, AttrTy::kDistribution,
                               &distribution};
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateRng(shape, *distribution, operands));
      break;
    }
    case HloOpcode::kRngGetAndUpdateState: {
      optional<int64> delta;
      attrs["delta"] = {/*required=*/true, AttrTy::kInt64, &delta};
      if (!ParseOperands(&operands, /*expected_size=*/0) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateRngGetAndUpdateState(shape, *delta));
      break;
    }
    case HloOpcode::kRngBitGenerator: {
      optional<RandomAlgorithm> algorithm;
      attrs["algorithm"] = {/*required=*/true, AttrTy::kRandomAlgorithm,
                            &algorithm};
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      instruction =
          builder->AddInstruction(HloInstruction::CreateRngBitGenerator(
              shape, operands[0], *algorithm));
      break;
    }
    case HloOpcode::kReducePrecision: {
      optional<int64> exponent_bits;
      optional<int64> mantissa_bits;
      attrs["exponent_bits"] = {/*required=*/true, AttrTy::kInt64,
                                &exponent_bits};
      attrs["mantissa_bits"] = {/*required=*/true, AttrTy::kInt64,
                                &mantissa_bits};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction =
          builder->AddInstruction(HloInstruction::CreateReducePrecision(
              shape, operands[0], static_cast<int>(*exponent_bits),
              static_cast<int>(*mantissa_bits)));
      break;
    }
    case HloOpcode::kConditional: {
      optional<HloComputation*> true_computation;
      optional<HloComputation*> false_computation;
      optional<std::vector<HloComputation*>> branch_computations;
      if (!ParseOperands(&operands)) {
        return false;
      }
      if (!ShapeUtil::IsScalar(operands[0]->shape())) {
        return Error(lexer_.GetLoc(), "The first operand must be a scalar");
      }
      const bool branch_index_is_bool =
          operands[0]->shape().element_type() == PRED;
      if (branch_index_is_bool) {
        attrs["true_computation"] = {/*required=*/true, AttrTy::kHloComputation,
                                     &true_computation};
        attrs["false_computation"] = {
            /*required=*/true, AttrTy::kHloComputation, &false_computation};
      } else {
        if (operands[0]->shape().element_type() != S32) {
          return Error(lexer_.GetLoc(),
                       "The first operand must be a scalar of PRED or S32");
        }
        attrs["branch_computations"] = {/*required=*/true,
                                        AttrTy::kBracedHloComputationList,
                                        &branch_computations};
      }
      if (!ParseAttributes(attrs)) {
        return false;
      }
      if (branch_index_is_bool) {
        branch_computations.emplace({*true_computation, *false_computation});
      }
      if (branch_computations->empty() ||
          operands.size() != branch_computations->size() + 1) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateConditional(
          shape, /*branch_index=*/operands[0],
          absl::MakeSpan(*branch_computations),
          absl::MakeSpan(operands).subspan(1)));
      break;
    }
    case HloOpcode::kCustomCall: {
      optional<std::string> custom_call_target;
      optional<Window> window;
      optional<ConvolutionDimensionNumbers> dnums;
      optional<int64> feature_group_count;
      optional<int64> batch_group_count;
      optional<std::vector<Shape>> operand_layout_constraints;
      optional<bool> custom_call_has_side_effect;
      optional<HloComputation*> to_apply;
      attrs["custom_call_target"] = {/*required=*/true, AttrTy::kString,
                                     &custom_call_target};
      attrs["window"] = {/*required=*/false, AttrTy::kWindow, &window};
      attrs["dim_labels"] = {/*required=*/false,
                             AttrTy::kConvolutionDimensionNumbers, &dnums};
      attrs["feature_group_count"] = {/*required=*/false, AttrTy::kInt64,
                                      &feature_group_count};
      attrs["batch_group_count"] = {/*required=*/false, AttrTy::kInt64,
                                    &batch_group_count};
      attrs["operand_layout_constraints"] = {
          /*required=*/false, AttrTy::kShapeList, &operand_layout_constraints};
      attrs["custom_call_has_side_effect"] = {/*required=*/false, AttrTy::kBool,
                                              &custom_call_has_side_effect};
      attrs["to_apply"] = {/*required=*/false, AttrTy::kHloComputation,
                           &to_apply};
      if (!ParseOperands(&operands) || !ParseAttributes(attrs)) {
        return false;
      }
      if (operand_layout_constraints.has_value()) {
        if (!LayoutUtil::HasLayout(shape)) {
          return Error(lexer_.GetLoc(),
                       "Layout must be set on layout-constrained custom call");
        }
        if (operands.size() != operand_layout_constraints->size()) {
          return Error(lexer_.GetLoc(),
                       StrCat("Expected ", operands.size(),
                              " operand layout constraints, ",
                              operand_layout_constraints->size(), " given"));
        }
        for (int64 i = 0; i < operands.size(); ++i) {
          const Shape& operand_shape_with_layout =
              (*operand_layout_constraints)[i];
          if (!LayoutUtil::HasLayout(operand_shape_with_layout)) {
            return Error(lexer_.GetLoc(),
                         StrCat("Operand layout constraint shape ",
                                ShapeUtil::HumanStringWithLayout(
                                    operand_shape_with_layout),
                                " for operand ", i, " does not have a layout"));
          }
          if (!ShapeUtil::Compatible(operand_shape_with_layout,
                                     operands[i]->shape())) {
            return Error(
                lexer_.GetLoc(),
                StrCat(
                    "Operand layout constraint shape ",
                    ShapeUtil::HumanStringWithLayout(operand_shape_with_layout),
                    " for operand ", i,
                    " is not compatible with operand shape ",
                    ShapeUtil::HumanStringWithLayout(operands[i]->shape())));
          }
        }
        instruction = builder->AddInstruction(HloInstruction::CreateCustomCall(
            shape, operands, *custom_call_target, *operand_layout_constraints,
            backend_config ? *backend_config : ""));
      } else {
        if (to_apply.has_value()) {
          instruction =
              builder->AddInstruction(HloInstruction::CreateCustomCall(
                  shape, operands, *to_apply, *custom_call_target,
                  backend_config ? *backend_config : ""));
        } else {
          instruction =
              builder->AddInstruction(HloInstruction::CreateCustomCall(
                  shape, operands, *custom_call_target,
                  backend_config ? *backend_config : ""));
        }
      }
      auto custom_call_instr = Cast<HloCustomCallInstruction>(instruction);
      if (window.has_value()) {
        custom_call_instr->set_window(*window);
      }
      if (dnums.has_value()) {
        custom_call_instr->set_convolution_dimension_numbers(*dnums);
      }
      if (feature_group_count.has_value()) {
        custom_call_instr->set_feature_group_count(*feature_group_count);
      }
      if (batch_group_count.has_value()) {
        custom_call_instr->set_batch_group_count(*batch_group_count);
      }
      if (custom_call_has_side_effect.has_value()) {
        custom_call_instr->set_custom_call_has_side_effect(
            *custom_call_has_side_effect);
      }
      break;
    }
    case HloOpcode::kDot: {
      optional<std::vector<int64>> lhs_contracting_dims;
      attrs["lhs_contracting_dims"] = {
          /*required=*/false, AttrTy::kBracedInt64List, &lhs_contracting_dims};
      optional<std::vector<int64>> rhs_contracting_dims;
      attrs["rhs_contracting_dims"] = {
          /*required=*/false, AttrTy::kBracedInt64List, &rhs_contracting_dims};
      optional<std::vector<int64>> lhs_batch_dims;
      attrs["lhs_batch_dims"] = {/*required=*/false, AttrTy::kBracedInt64List,
                                 &lhs_batch_dims};
      optional<std::vector<int64>> rhs_batch_dims;
      attrs["rhs_batch_dims"] = {/*required=*/false, AttrTy::kBracedInt64List,
                                 &rhs_batch_dims};
      optional<std::vector<PrecisionConfig::Precision>> operand_precision;
      attrs["operand_precision"] = {/*required=*/false, AttrTy::kPrecisionList,
                                    &operand_precision};

      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributes(attrs)) {
        return false;
      }

      DotDimensionNumbers dnum;
      if (lhs_contracting_dims) {
        *dnum.mutable_lhs_contracting_dimensions() = {
            lhs_contracting_dims->begin(), lhs_contracting_dims->end()};
      }
      if (rhs_contracting_dims) {
        *dnum.mutable_rhs_contracting_dimensions() = {
            rhs_contracting_dims->begin(), rhs_contracting_dims->end()};
      }
      if (lhs_batch_dims) {
        *dnum.mutable_lhs_batch_dimensions() = {lhs_batch_dims->begin(),
                                                lhs_batch_dims->end()};
      }
      if (rhs_batch_dims) {
        *dnum.mutable_rhs_batch_dimensions() = {rhs_batch_dims->begin(),
                                                rhs_batch_dims->end()};
      }

      PrecisionConfig precision_config;
      if (operand_precision) {
        *precision_config.mutable_operand_precision() = {
            operand_precision->begin(), operand_precision->end()};
      } else {
        precision_config.mutable_operand_precision()->Resize(
            operands.size(), PrecisionConfig::DEFAULT);
      }

      instruction = builder->AddInstruction(HloInstruction::CreateDot(
          shape, operands[0], operands[1], dnum, precision_config));
      break;
    }
    case HloOpcode::kGather: {
      optional<std::vector<int64>> offset_dims;
      attrs["offset_dims"] = {/*required=*/true, AttrTy::kBracedInt64List,
                              &offset_dims};
      optional<std::vector<int64>> collapsed_slice_dims;
      attrs["collapsed_slice_dims"] = {
          /*required=*/true, AttrTy::kBracedInt64List, &collapsed_slice_dims};
      optional<std::vector<int64>> start_index_map;
      attrs["start_index_map"] = {/*required=*/true, AttrTy::kBracedInt64List,
                                  &start_index_map};
      optional<int64> index_vector_dim;
      attrs["index_vector_dim"] = {/*required=*/true, AttrTy::kInt64,
                                   &index_vector_dim};
      optional<std::vector<int64>> slice_sizes;
      attrs["slice_sizes"] = {/*required=*/true, AttrTy::kBracedInt64List,
                              &slice_sizes};
      optional<bool> indices_are_sorted = false;
      attrs["indices_are_sorted"] = {/*required=*/false, AttrTy::kBool,
                                     &indices_are_sorted};

      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributes(attrs)) {
        return false;
      }

      GatherDimensionNumbers dim_numbers =
          HloGatherInstruction::MakeGatherDimNumbers(
              /*offset_dims=*/*offset_dims,
              /*collapsed_slice_dims=*/*collapsed_slice_dims,
              /*start_index_map=*/*start_index_map,
              /*index_vector_dim=*/*index_vector_dim);

      instruction = builder->AddInstruction(HloInstruction::CreateGather(
          shape, /*operand=*/operands[0], /*start_indices=*/operands[1],
          dim_numbers, *slice_sizes, indices_are_sorted.value()));
      break;
    }
    case HloOpcode::kScatter: {
      optional<std::vector<int64>> update_window_dims;
      attrs["update_window_dims"] = {
          /*required=*/true, AttrTy::kBracedInt64List, &update_window_dims};
      optional<std::vector<int64>> inserted_window_dims;
      attrs["inserted_window_dims"] = {
          /*required=*/true, AttrTy::kBracedInt64List, &inserted_window_dims};
      optional<std::vector<int64>> scatter_dims_to_operand_dims;
      attrs["scatter_dims_to_operand_dims"] = {/*required=*/true,
                                               AttrTy::kBracedInt64List,
                                               &scatter_dims_to_operand_dims};
      optional<int64> index_vector_dim;
      attrs["index_vector_dim"] = {/*required=*/true, AttrTy::kInt64,
                                   &index_vector_dim};

      optional<HloComputation*> update_computation;
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &update_computation};
      optional<bool> indices_are_sorted = false;
      attrs["indices_are_sorted"] = {/*required=*/false, AttrTy::kBool,
                                     &indices_are_sorted};
      optional<bool> unique_indices = false;
      attrs["unique_indices"] = {/*required=*/false, AttrTy::kBool,
                                 &unique_indices};

      if (!ParseOperands(&operands, /*expected_size=*/3) ||
          !ParseAttributes(attrs)) {
        return false;
      }

      ScatterDimensionNumbers dim_numbers =
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/*update_window_dims,
              /*inserted_window_dims=*/*inserted_window_dims,
              /*scatter_dims_to_operand_dims=*/*scatter_dims_to_operand_dims,
              /*index_vector_dim=*/*index_vector_dim);

      instruction = builder->AddInstruction(HloInstruction::CreateScatter(
          shape, /*operand=*/operands[0], /*scatter_indices=*/operands[1],
          /*updates=*/operands[2], *update_computation, dim_numbers,
          indices_are_sorted.value(), unique_indices.value()));
      break;
    }
    case HloOpcode::kDomain: {
      DomainData domain;
      attrs["domain"] = {/*required=*/true, AttrTy::kDomain, &domain};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateDomain(
          shape, operands[0], std::move(domain.exit_metadata),
          std::move(domain.entry_metadata)));
      break;
    }
    case HloOpcode::kTrace:
      return TokenError(StrCat("parsing not yet implemented for op: ",
                               HloOpcodeString(opcode)));
    case HloOpcode::kGetDimensionSize: {
      optional<std::vector<int64>> dimensions;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions};
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction =
          builder->AddInstruction(HloInstruction::CreateGetDimensionSize(
              shape, operands[0], (*dimensions)[0]));
      break;
    }
    case HloOpcode::kSetDimensionSize: {
      optional<std::vector<int64>> dimensions;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions};
      if (!ParseOperands(&operands, /*expected_size=*/2) ||
          !ParseAttributes(attrs)) {
        return false;
      }
      instruction =
          builder->AddInstruction(HloInstruction::CreateSetDimensionSize(
              shape, operands[0], operands[1], (*dimensions)[0]));
      break;
    }
  }

  instruction->SetAndSanitizeName(name);
  if (instruction->name() != name) {
    return Error(name_loc,
                 StrCat("illegal instruction name: ", name,
                        "; suggest renaming to: ", instruction->name()));
  }

  // Add shared attributes like metadata to the instruction, if they were seen.
  if (sharding) {
    instruction->set_sharding(
        HloSharding::FromProto(sharding.value()).ValueOrDie());
  }
  if (parameter_replication) {
    int leaf_count = ShapeUtil::GetLeafCount(instruction->shape());
    const auto& replicated =
        parameter_replication->replicated_at_leaf_buffers();
    if (leaf_count != replicated.size()) {
      return Error(lexer_.GetLoc(),
                   StrCat("parameter has ", leaf_count,
                          " leaf buffers, but parameter_replication has ",
                          replicated.size(), " elements."));
    }
    instruction->set_parameter_replicated_at_leaf_buffers(replicated);
  }
  if (predecessors) {
    for (auto* pre : *predecessors) {
      Status status = pre->AddControlDependencyTo(instruction);
      if (!status.ok()) {
        return Error(name_loc, StrCat("error adding control dependency for: ",
                                      name, " status: ", status.ToString()));
      }
    }
  }
  if (metadata) {
    instruction->set_metadata(*metadata);
  }
  if (backend_config) {
    instruction->set_raw_backend_config_string(std::move(*backend_config));
  }
  if (outer_dimension_partitions) {
    instruction->set_outer_dimension_partitions(*outer_dimension_partitions);
  }
  if (frontend_attributes) {
    instruction->set_frontend_attributes(*frontend_attributes);
  }
  return AddInstruction(name, instruction, name_loc);
}  // NOLINT(readability/fn_size)

// ::= '{' (single_sharding | tuple_sharding) '}'
//
// tuple_sharding ::= single_sharding* (',' single_sharding)*
bool HloParserImpl::ParseSharding(OpSharding* sharding) {
  // A single sharding starts with '{' and is not followed by '{'.
  // A tuple sharding starts with '{' and is followed by '{', or is '{''}' for
  // an empty tuple.
  if (!ParseToken(TokKind::kLbrace,
                  "expected '{' to start sharding attribute")) {
    return false;
  }

  if (lexer_.GetKind() != TokKind::kLbrace &&
      lexer_.GetKind() != TokKind::kRbrace) {
    return ParseSingleSharding(sharding, /*lbrace_pre_lexed=*/true);
  }

  // Tuple sharding.
  // Allow empty tuple shardings.
  if (lexer_.GetKind() != TokKind::kRbrace) {
    do {
      if (!ParseSingleSharding(sharding->add_tuple_shardings(),
                               /*lbrace_pre_lexed=*/false)) {
        return false;
      }
    } while (EatIfPresent(TokKind::kComma));
  }
  sharding->set_type(OpSharding::TUPLE);

  return ParseToken(TokKind::kRbrace, "expected '}' to end sharding attribute");
}

// frontend_attributes ::= '{' attributes '}'
// attributes
//   ::= /*empty*/
//   ::= attribute '=' value (',' attribute '=' value)*
bool HloParserImpl::ParseFrontendAttributes(
    FrontendAttributes* frontend_attributes) {
  CHECK(frontend_attributes != nullptr);
  if (!ParseToken(TokKind::kLbrace,
                  "expected '{' to start frontend attributes")) {
    return false;
  }
  if (lexer_.GetKind() == TokKind::kRbrace) {
    // empty
  } else {
    do {
      std::string attribute;
      if (!ParseAttributeName(&attribute)) {
        return false;
      }
      if (lexer_.GetKind() != TokKind::kString) {
        return false;
      }
      (*frontend_attributes->mutable_map())[attribute] = lexer_.GetStrVal();
      lexer_.Lex();
    } while (EatIfPresent(TokKind::kComma));
  }
  return ParseToken(TokKind::kRbrace,
                    "expects '}' at the end of frontend attributes");
}

//  ::= '{' 'replicated'? 'maximal'? ('device=' int)? shape?
//          ('devices=' ('[' dims ']')* device_list)? '}'
// dims ::= int_list device_list ::= int_list
bool HloParserImpl::ParseSingleSharding(OpSharding* sharding,
                                        bool lbrace_pre_lexed) {
  if (!lbrace_pre_lexed &&
      !ParseToken(TokKind::kLbrace,
                  "expected '{' to start sharding attribute")) {
    return false;
  }

  LocTy loc = lexer_.GetLoc();
  bool maximal = false;
  bool replicated = false;
  std::vector<int64> devices;
  std::vector<int64> tile_assignment_dimensions;
  while (lexer_.GetKind() != TokKind::kRbrace) {
    switch (lexer_.GetKind()) {
      case TokKind::kw_maximal:
        maximal = true;
        lexer_.Lex();
        break;
      case TokKind::kw_replicated:
        replicated = true;
        lexer_.Lex();
        break;
      case TokKind::kAttributeName: {
        if (lexer_.GetStrVal() == "device") {
          if (lexer_.Lex() != TokKind::kInt) {
            return TokenError("device= attribute must be an integer");
          }
          devices = {lexer_.GetInt64Val()};
          lexer_.Lex();
        } else if (lexer_.GetStrVal() == "devices") {
          lexer_.Lex();
          if (!ParseToken(TokKind::kLsquare,
                          "expected '[' to start sharding devices shape")) {
            return false;
          }

          do {
            int64 dim;
            if (!ParseInt64(&dim)) {
              return false;
            }
            tile_assignment_dimensions.push_back(dim);
          } while (EatIfPresent(TokKind::kComma));

          if (!ParseToken(TokKind::kRsquare,
                          "expected ']' to start sharding devices shape")) {
            return false;
          }
          do {
            int64 device;
            if (!ParseInt64(&device)) {
              return false;
            }
            devices.push_back(device);
          } while (EatIfPresent(TokKind::kComma));
        } else {
          return TokenError(
              "unknown attribute in sharding: expected device= or devices=");
        }
        break;
      }
      case TokKind::kRbrace:
        break;
      default:
        return TokenError("unexpected token");
    }
  }

  if (replicated) {
    if (!devices.empty()) {
      return Error(loc,
                   "replicated shardings should not have any devices assigned");
    }
    sharding->set_type(OpSharding::REPLICATED);
  } else if (maximal) {
    if (devices.size() != 1) {
      return Error(loc,
                   "maximal shardings should have exactly one device assigned");
    }
    sharding->set_type(OpSharding::MAXIMAL);
    sharding->add_tile_assignment_devices(devices[0]);
  } else {
    if (devices.size() <= 1) {
      return Error(
          loc, "non-maximal shardings must have more than one device assigned");
    }
    if (tile_assignment_dimensions.empty()) {
      return Error(
          loc,
          "non-maximal shardings must have a tile assignment list including "
          "dimensions");
    }
    sharding->set_type(OpSharding::OTHER);
    for (int64 dim : tile_assignment_dimensions) {
      sharding->add_tile_assignment_dimensions(dim);
    }
    for (int64 device : devices) {
      sharding->add_tile_assignment_devices(device);
    }
  }

  lexer_.Lex();
  return true;
}

// parameter_replication ::=
//   '{' ('true' | 'false')* (',' ('true' | 'false'))*  '}'
bool HloParserImpl::ParseParameterReplication(
    ParameterReplication* parameter_replication) {
  if (!ParseToken(TokKind::kLbrace,
                  "expected '{' to start parameter_replication attribute")) {
    return false;
  }

  if (lexer_.GetKind() != TokKind::kRbrace) {
    do {
      if (lexer_.GetKind() == TokKind::kw_true) {
        parameter_replication->add_replicated_at_leaf_buffers(true);
      } else if (lexer_.GetKind() == TokKind::kw_false) {
        parameter_replication->add_replicated_at_leaf_buffers(false);
      } else {
        return false;
      }
      lexer_.Lex();
    } while (EatIfPresent(TokKind::kComma));
  }

  return ParseToken(TokKind::kRbrace,
                    "expected '}' to end parameter_replication attribute");
}

// replica_groups ::='{' int64list_elements '}'
// int64list_elements
//   ::= /*empty*/
//   ::= int64list (',' int64list)*
// int64list ::= '{' int64_elements '}'
// int64_elements
//   ::= /*empty*/
//   ::= int64_val (',' int64_val)*
bool HloParserImpl::ParseReplicaGroupsOnly(
    std::vector<ReplicaGroup>* replica_groups) {
  std::vector<std::vector<int64>> result;
  if (!ParseInt64ListList(TokKind::kLbrace, TokKind::kRbrace, TokKind::kComma,
                          &result)) {
    return false;
  }
  *replica_groups = CreateReplicaGroups(result);
  return true;
}

// domain ::= '{' 'kind=' domain_kind ',' 'entry=' entry_sharding ','
//            'exit=' exit_sharding '}'
bool HloParserImpl::ParseDomain(DomainData* domain) {
  absl::flat_hash_map<std::string, AttrConfig> attrs;
  optional<std::string> kind;
  optional<OpSharding> entry_sharding;
  optional<OpSharding> exit_sharding;
  attrs["kind"] = {/*required=*/true, AttrTy::kString, &kind};
  attrs["entry"] = {/*required=*/true, AttrTy::kSharding, &entry_sharding};
  attrs["exit"] = {/*required=*/true, AttrTy::kSharding, &exit_sharding};
  if (!ParseSubAttributes(attrs)) {
    return false;
  }
  if (*kind == ShardingMetadata::KindName()) {
    auto entry_sharding_ptr = absl::make_unique<HloSharding>(
        HloSharding::FromProto(*entry_sharding).ValueOrDie());
    auto exit_sharding_ptr = absl::make_unique<HloSharding>(
        HloSharding::FromProto(*exit_sharding).ValueOrDie());
    domain->entry_metadata =
        absl::make_unique<ShardingMetadata>(std::move(entry_sharding_ptr));
    domain->exit_metadata =
        absl::make_unique<ShardingMetadata>(std::move(exit_sharding_ptr));
  } else {
    return TokenError(StrCat("unsupported domain kind: ", *kind));
  }
  return true;
}

// '{' name+ '}'
bool HloParserImpl::ParseInstructionNames(
    std::vector<HloInstruction*>* instructions) {
  if (!ParseToken(TokKind::kLbrace,
                  "expects '{' at the beginning of instruction name list")) {
    return false;
  }
  LocTy loc = lexer_.GetLoc();
  do {
    std::string name;
    if (!ParseName(&name)) {
      return Error(loc, "expects a instruction name");
    }
    std::pair<HloInstruction*, LocTy>* instr = FindInstruction(name);
    if (!instr) {
      return TokenError(StrFormat("instruction '%s' is not defined", name));
    }
    instructions->push_back(instr->first);
  } while (EatIfPresent(TokKind::kComma));

  return ParseToken(TokKind::kRbrace,
                    "expects '}' at the end of instruction name list");
}

bool HloParserImpl::SetValueInLiteral(LocTy loc, int64 value, int64 index,
                                      Literal* literal) {
  const Shape& shape = literal->shape();
  switch (shape.element_type()) {
    case S8:
      return SetValueInLiteralHelper<int8>(loc, value, index, literal);
    case S16:
      return SetValueInLiteralHelper<int16>(loc, value, index, literal);
    case S32:
      return SetValueInLiteralHelper<int32>(loc, value, index, literal);
    case S64:
      return SetValueInLiteralHelper<int64>(loc, value, index, literal);
    case U8:
      return SetValueInLiteralHelper<tensorflow::uint8>(loc, value, index,
                                                        literal);
    case U16:
      return SetValueInLiteralHelper<tensorflow::uint16>(loc, value, index,
                                                         literal);
    case U32:
      return SetValueInLiteralHelper<tensorflow::uint32>(loc, value, index,
                                                         literal);
    case U64:
      return SetValueInLiteralHelper<tensorflow::uint64>(loc, value, index,
                                                         literal);
    case PRED:
      // Bool type literals with rank >= 1 are printed in 0s and 1s.
      return SetValueInLiteralHelper<bool>(loc, static_cast<bool>(value), index,
                                           literal);
    default:
      LOG(FATAL) << "unknown integral primitive type "
                 << PrimitiveType_Name(shape.element_type());
  }
}

bool HloParserImpl::SetValueInLiteral(LocTy loc, double value, int64 index,
                                      Literal* literal) {
  const Shape& shape = literal->shape();
  switch (shape.element_type()) {
    case F16:
      return SetValueInLiteralHelper<Eigen::half>(loc, value, index, literal);
    case BF16:
      return SetValueInLiteralHelper<tensorflow::bfloat16>(loc, value, index,
                                                           literal);
    case F32:
      return SetValueInLiteralHelper<float>(loc, value, index, literal);
    case F64:
      return SetValueInLiteralHelper<double>(loc, value, index, literal);
    default:
      LOG(FATAL) << "unknown floating point primitive type "
                 << PrimitiveType_Name(shape.element_type());
  }
}

bool HloParserImpl::SetValueInLiteral(LocTy loc, bool value, int64 index,
                                      Literal* literal) {
  const Shape& shape = literal->shape();
  switch (shape.element_type()) {
    case PRED:
      return SetValueInLiteralHelper<bool>(loc, value, index, literal);
    default:
      LOG(FATAL) << PrimitiveType_Name(shape.element_type())
                 << " is not PRED type";
  }
}

bool HloParserImpl::SetValueInLiteral(LocTy loc, std::complex<double> value,
                                      int64 index, Literal* literal) {
  const Shape& shape = literal->shape();
  switch (shape.element_type()) {
    case C64:
      return SetValueInLiteralHelper<std::complex<float>>(loc, value, index,
                                                          literal);
    case C128:
      return SetValueInLiteralHelper<std::complex<double>>(loc, value, index,
                                                           literal);
    default:
      LOG(FATAL) << PrimitiveType_Name(shape.element_type())
                 << " is not a complex type type";
  }
}

template <typename T>
std::string StringifyValue(T val) {
  return StrCat(val);
}
template <>
std::string StringifyValue(std::complex<double> val) {
  return StrFormat("(%f, %f)", std::real(val), std::imag(val));
}

template <typename LiteralNativeT, typename ParsedElemT>
bool HloParserImpl::SetValueInLiteralHelper(LocTy loc, ParsedElemT value,
                                            int64 index, Literal* literal) {
  if (!CheckParsedValueIsInRange<LiteralNativeT>(loc, value)) {
    return false;
  }

  // Check that the index is in range and assign into the literal
  if (index >= ShapeUtil::ElementsIn(literal->shape())) {
    return Error(loc, StrCat("trys to set value ", StringifyValue(value),
                             " to a literal in shape ",
                             ShapeUtil::HumanString(literal->shape()),
                             " at linear index ", index,
                             ", but the index is out of range"));
  }
  literal->data<LiteralNativeT>().at(index) =
      static_cast<LiteralNativeT>(value);
  return true;
}

// literal
//  ::= tuple
//  ::= non_tuple
bool HloParserImpl::ParseLiteral(Literal* literal, const Shape& shape) {
  return shape.IsTuple() ? ParseTupleLiteral(literal, shape)
                         : ParseNonTupleLiteral(literal, shape);
}

// tuple
//  ::= shape '(' literal_list ')'
// literal_list
//  ::= /*empty*/
//  ::= literal (',' literal)*
bool HloParserImpl::ParseTupleLiteral(Literal* literal, const Shape& shape) {
  if (!ParseToken(TokKind::kLparen, "expects '(' in front of tuple elements")) {
    return false;
  }
  std::vector<Literal> elements(ShapeUtil::TupleElementCount(shape));

  if (lexer_.GetKind() == TokKind::kRparen) {
    // empty
  } else {
    // literal, (',' literal)*
    for (int i = 0; i < elements.size(); i++) {
      if (i > 0) {
        ParseToken(TokKind::kComma, "expects ',' to separate tuple elements");
      }
      if (!ParseLiteral(&elements[i],
                        ShapeUtil::GetTupleElementShape(shape, i))) {
        return TokenError(StrCat("expects the ", i, "th element"));
      }
    }
  }
  *literal = LiteralUtil::MakeTupleOwned(std::move(elements));
  return ParseToken(TokKind::kRparen,
                    StrCat("expects ')' at the end of the tuple with ",
                           ShapeUtil::TupleElementCount(shape), "elements"));
}

// non_tuple
//   ::= rank01
//   ::= rank2345
// rank2345 ::= shape nested_array
bool HloParserImpl::ParseNonTupleLiteral(Literal* literal, const Shape& shape) {
  CHECK(LayoutUtil::IsDenseArray(shape)) << shape.ToString(true);
  return ParseDenseLiteral(literal, shape);
}

bool HloParserImpl::ParseDenseLiteral(Literal* literal, const Shape& shape) {
  // Cast `rank` to int because we call shape.dimensions(int rank) below, and if
  // `rank` is an int64, that's an implicit narrowing conversion, which is
  // implementation-defined behavior.
  const int rank = static_cast<int>(shape.rank());

  // Create a literal with the given shape in default layout.
  *literal = LiteralUtil::CreateFromDimensions(
      shape.element_type(), AsInt64Slice(shape.dimensions()));
  int64 nest_level = 0;
  int64 linear_index = 0;
  // elems_seen_per_dim[i] is how many elements or sub-arrays we have seen for
  // the dimension i. For example, to parse f32[2,3] {{1, 2, 3}, {4, 5, 6}},
  // when we are parsing the 2nd '{' (right before '1'), we are seeing a
  // sub-array of the dimension 0, so elems_seen_per_dim[0]++. When we are at
  // the first '}' (right after '3'), it means the sub-array ends, and the
  // sub-array is supposed to contain exactly 3 elements, so check if
  // elems_seen_per_dim[1] is 3.
  std::vector<int64> elems_seen_per_dim(rank);
  auto get_index_str = [&elems_seen_per_dim](int dim) -> std::string {
    std::vector<int64> elems_seen_until_dim(elems_seen_per_dim.begin(),
                                            elems_seen_per_dim.begin() + dim);
    return StrCat("[",
                  StrJoin(elems_seen_until_dim, ",",
                          [](std::string* out, const int64 num_elems) {
                            StrAppend(out, num_elems - 1);
                          }),
                  "]");
  };

  auto add_one_elem_seen = [&] {
    if (rank > 0) {
      if (nest_level != rank) {
        return TokenError(absl::StrFormat(
            "expects nested array in rank %d, but sees %d", rank, nest_level));
      }
      elems_seen_per_dim[rank - 1]++;
      if (elems_seen_per_dim[rank - 1] > shape.dimensions(rank - 1)) {
        return TokenError(absl::StrFormat(
            "expects %d elements on the minor-most dimension, but "
            "sees more",
            shape.dimensions(rank - 1)));
      }
    }
    return true;
  };

  do {
    switch (lexer_.GetKind()) {
      default:
        return TokenError("unexpected token type in a literal");
      case TokKind::kLbrace: {
        nest_level++;
        if (nest_level > rank) {
          return TokenError(absl::StrFormat(
              "expects nested array in rank %d, but sees larger", rank));
        }
        if (nest_level > 1) {
          elems_seen_per_dim[nest_level - 2]++;
          if (elems_seen_per_dim[nest_level - 2] >
              shape.dimensions(nest_level - 2)) {
            return TokenError(absl::StrFormat(
                "expects %d elements in the %sth element, but sees more",
                shape.dimensions(nest_level - 2),
                get_index_str(nest_level - 2)));
          }
        }
        lexer_.Lex();
        break;
      }
      case TokKind::kRbrace: {
        nest_level--;
        if (elems_seen_per_dim[nest_level] != shape.dimensions(nest_level)) {
          return TokenError(absl::StrFormat(
              "expects %d elements in the %sth element, but sees %d",
              shape.dimensions(nest_level), get_index_str(nest_level),
              elems_seen_per_dim[nest_level]));
        }
        elems_seen_per_dim[nest_level] = 0;
        lexer_.Lex();
        break;
      }
      case TokKind::kLparen: {
        if (!primitive_util::IsComplexType(shape.element_type())) {
          return TokenError(
              absl::StrFormat("unexpected '(' in literal.  Parens are only "
                              "valid for complex literals"));
        }

        std::complex<double> value;
        LocTy loc = lexer_.GetLoc();
        if (!add_one_elem_seen() || !ParseComplex(&value) ||
            !SetValueInLiteral(loc, value, linear_index++, literal)) {
          return false;
        }
        break;
      }
      case TokKind::kDots: {
        if (nest_level != 1) {
          return TokenError(absl::StrFormat(
              "expects `...` at nest level 1, but sees it at nest level %d",
              nest_level));
        }
        elems_seen_per_dim[0] = shape.dimensions(0);
        lexer_.Lex();
        // Fill data with deterministic (garbage) values. Use static to avoid
        // creating identical constants which could potentially got CSE'ed
        // away. This is a best-effort approach to make sure replaying a HLO
        // gives us same optimized HLO graph.
        static uint32 data = 0;
        uint32* raw_data = static_cast<uint32*>(literal->untyped_data());
        for (int64 i = 0; i < literal->size_bytes() / 4; ++i) {
          raw_data[i] = data++;
        }
        uint8* raw_data_int8 = static_cast<uint8*>(literal->untyped_data());
        static uint8 data_int8 = 0;
        for (int64 i = 0; i < literal->size_bytes() % 4; ++i) {
          raw_data_int8[literal->size_bytes() / 4 + i] = data_int8++;
        }
        break;
      }
      case TokKind::kComma:
        // Skip.
        lexer_.Lex();
        break;
      case TokKind::kw_true:
      case TokKind::kw_false:
      case TokKind::kInt:
      case TokKind::kDecimal:
      case TokKind::kw_nan:
      case TokKind::kw_inf:
      case TokKind::kNegInf: {
        add_one_elem_seen();
        if (lexer_.GetKind() == TokKind::kw_true ||
            lexer_.GetKind() == TokKind::kw_false) {
          if (!SetValueInLiteral(lexer_.GetLoc(),
                                 lexer_.GetKind() == TokKind::kw_true,
                                 linear_index++, literal)) {
            return false;
          }
          lexer_.Lex();
        } else if (primitive_util::IsIntegralType(shape.element_type()) ||
                   shape.element_type() == PRED) {
          LocTy loc = lexer_.GetLoc();
          int64 value;
          if (!ParseInt64(&value)) {
            return Error(loc, StrCat("expects integer for primitive type: ",
                                     PrimitiveType_Name(shape.element_type())));
          }
          if (!SetValueInLiteral(loc, value, linear_index++, literal)) {
            return false;
          }
        } else if (primitive_util::IsFloatingPointType(shape.element_type())) {
          LocTy loc = lexer_.GetLoc();
          double value;
          if (!ParseDouble(&value)) {
            return Error(
                loc, StrCat("expect floating point value for primitive type: ",
                            PrimitiveType_Name(shape.element_type())));
          }
          if (!SetValueInLiteral(loc, value, linear_index++, literal)) {
            return false;
          }
        } else {
          return TokenError(StrCat("unsupported primitive type ",
                                   PrimitiveType_Name(shape.element_type())));
        }
        break;
      }
    }  // end of switch
  } while (nest_level > 0);

  *literal = literal->Relayout(shape.layout());
  return true;
}

// MaxFiniteValue is a type-traits helper used by
// HloParserImpl::CheckParsedValueIsInRange.
template <typename T>
struct MinMaxFiniteValue {
  static T max() { return std::numeric_limits<T>::max(); }
  static T min() { return std::numeric_limits<T>::lowest(); }
};

template <>
struct MinMaxFiniteValue<Eigen::half> {
  static double max() {
    // Sadly this is not constexpr, so this forces `value` to be a method.
    return static_cast<double>(Eigen::NumTraits<Eigen::half>::highest());
  }
  static double min() { return -max(); }
};

template <>
struct MinMaxFiniteValue<bfloat16> {
  static double max() { return static_cast<double>(bfloat16::highest()); }
  static double min() { return -max(); }
};

// MSVC's standard C++ library does not define isnan/isfinite for integer types.
// To work around that we will need to provide our own.
template <typename T>
std::enable_if_t<std::is_floating_point<T>::value, bool> IsFinite(T val) {
  return std::isfinite(val);
}
template <typename T>
std::enable_if_t<std::is_floating_point<T>::value, bool> IsNaN(T val) {
  return std::isnan(val);
}
template <typename T>
std::enable_if_t<std::is_integral<T>::value, bool> IsFinite(T val) {
  return std::isfinite(static_cast<double>(val));
}
template <typename T>
std::enable_if_t<std::is_integral<T>::value, bool> IsNaN(T val) {
  return std::isnan(static_cast<double>(val));
}

template <typename LiteralNativeT, typename ParsedElemT>
bool HloParserImpl::CheckParsedValueIsInRange(LocTy loc, ParsedElemT value) {
  if (std::is_floating_point<ParsedElemT>::value) {
    auto value_as_native_t = static_cast<LiteralNativeT>(value);
    auto value_double_converted = static_cast<ParsedElemT>(value_as_native_t);
    if (!IsFinite(value) || IsFinite(value_double_converted)) {
      value = value_double_converted;
    }
  }
  PrimitiveType literal_ty =
      primitive_util::NativeToPrimitiveType<LiteralNativeT>();
  if (IsNaN(value) ||
      (std::numeric_limits<ParsedElemT>::has_infinity &&
       (std::numeric_limits<ParsedElemT>::infinity() == value ||
        -std::numeric_limits<ParsedElemT>::infinity() == value))) {
    // Skip range checking for non-finite value.
  } else if (std::is_unsigned<LiteralNativeT>::value) {
    CHECK((std::is_same<ParsedElemT, int64>::value ||
           std::is_same<ParsedElemT, bool>::value))
        << "Unimplemented checking for ParsedElemT";

    const uint64 unsigned_value = value;
    const uint64 upper_bound =
        static_cast<uint64>(std::numeric_limits<LiteralNativeT>::max());
    if (unsigned_value > upper_bound) {
      // Value is out of range for LiteralNativeT.
      return Error(loc, StrCat("value ", value,
                               " is out of range for literal's primitive type ",
                               PrimitiveType_Name(literal_ty), " namely [0, ",
                               upper_bound, "]."));
    }
  } else if (value > MinMaxFiniteValue<LiteralNativeT>::max() ||
             value < MinMaxFiniteValue<LiteralNativeT>::min()) {
    // Value is out of range for LiteralNativeT.
    return Error(loc, StrCat("value ", value,
                             " is out of range for literal's primitive type ",
                             PrimitiveType_Name(literal_ty), " namely [",
                             MinMaxFiniteValue<LiteralNativeT>::min(), ", ",
                             MinMaxFiniteValue<LiteralNativeT>::max(), "]."));
  }
  return true;
}

template <typename LiteralNativeT>
bool HloParserImpl::CheckParsedValueIsInRange(LocTy loc,
                                              std::complex<double> value) {
  // e.g. `float` for std::complex<float>
  using LiteralComplexComponentT =
      decltype(std::real(std::declval<LiteralNativeT>()));

  // We could do simply
  //
  //   return CheckParsedValueIsInRange<LiteralNativeT>(std::real(value)) &&
  //          CheckParsedValueIsInRange<LiteralNativeT>(std::imag(value));
  //
  // but this would give bad error messages on failure.

  auto check_component = [&](absl::string_view name, double v) {
    if (std::isnan(v) || v == std::numeric_limits<double>::infinity() ||
        v == -std::numeric_limits<double>::infinity()) {
      // Skip range-checking for non-finite values.
      return true;
    }

    double min = MinMaxFiniteValue<LiteralComplexComponentT>::min();
    double max = MinMaxFiniteValue<LiteralComplexComponentT>::max();
    if (v < min || v > max) {
      // Value is out of range for LitearlComplexComponentT.
      return Error(
          loc,
          StrCat(name, " part ", v,
                 " is out of range for literal's primitive type ",
                 PrimitiveType_Name(
                     primitive_util::NativeToPrimitiveType<LiteralNativeT>()),
                 ", namely [", min, ", ", max, "]."));
    }
    return true;
  };
  return check_component("real", std::real(value)) &&
         check_component("imaginary", std::imag(value));
}

// operands ::= '(' operands1 ')'
// operands1
//   ::= /*empty*/
//   ::= operand (, operand)*
// operand ::= (shape)? name
bool HloParserImpl::ParseOperands(std::vector<HloInstruction*>* operands) {
  CHECK(operands != nullptr);
  if (!ParseToken(TokKind::kLparen,
                  "expects '(' at the beginning of operands")) {
    return false;
  }
  if (lexer_.GetKind() == TokKind::kRparen) {
    // empty
  } else {
    do {
      LocTy loc = lexer_.GetLoc();
      std::string name;
      optional<Shape> shape;
      if (CanBeShape()) {
        shape.emplace();
        if (!ParseShape(&shape.value())) {
          return false;
        }
      }
      if (!ParseName(&name)) {
        // When parsing a single instruction (as opposed to a whole module), an
        // HLO may have one or more operands with a shape but no name:
        //
        //  foo = add(f32[10], f32[10])
        //
        // create_missing_instruction_ is always non-null when parsing a single
        // instruction, and is responsible for creating kParameter instructions
        // for these operands.
        if (shape.has_value() && create_missing_instruction_ != nullptr &&
            scoped_name_tables_.size() == 1) {
          name = "";
        } else {
          return false;
        }
      }
      std::pair<HloInstruction*, LocTy>* instruction =
          FindInstruction(name, shape);
      if (instruction == nullptr) {
        return Error(loc, StrCat("instruction does not exist: ", name));
      }
      operands->push_back(instruction->first);
    } while (EatIfPresent(TokKind::kComma));
  }
  return ParseToken(TokKind::kRparen, "expects ')' at the end of operands");
}

bool HloParserImpl::ParseOperands(std::vector<HloInstruction*>* operands,
                                  const int expected_size) {
  CHECK(operands != nullptr);
  LocTy loc = lexer_.GetLoc();
  if (!ParseOperands(operands)) {
    return false;
  }
  if (expected_size != operands->size()) {
    return Error(loc, StrCat("expects ", expected_size, " operands, but has ",
                             operands->size(), " operands"));
  }
  return true;
}

// sub_attributes ::= '{' (','? attribute)* '}'
bool HloParserImpl::ParseSubAttributes(
    const absl::flat_hash_map<std::string, AttrConfig>& attrs) {
  LocTy loc = lexer_.GetLoc();
  if (!ParseToken(TokKind::kLbrace, "expects '{' to start sub attributes")) {
    return false;
  }
  absl::flat_hash_set<std::string> seen_attrs;
  if (lexer_.GetKind() == TokKind::kRbrace) {
    // empty
  } else {
    do {
      EatIfPresent(TokKind::kComma);
      if (!ParseAttributeHelper(attrs, &seen_attrs)) {
        return false;
      }
    } while (lexer_.GetKind() != TokKind::kRbrace);
  }
  // Check that all required attrs were seen.
  for (const auto& attr_it : attrs) {
    if (attr_it.second.required &&
        seen_attrs.find(attr_it.first) == seen_attrs.end()) {
      return Error(loc, StrFormat("sub-attribute %s is expected but not seen",
                                  attr_it.first));
    }
  }
  return ParseToken(TokKind::kRbrace, "expects '}' to end sub attributes");
}

// attributes ::= (',' attribute)*
bool HloParserImpl::ParseAttributes(
    const absl::flat_hash_map<std::string, AttrConfig>& attrs) {
  LocTy loc = lexer_.GetLoc();
  absl::flat_hash_set<std::string> seen_attrs;
  while (EatIfPresent(TokKind::kComma)) {
    if (!ParseAttributeHelper(attrs, &seen_attrs)) {
      return false;
    }
  }
  // Check that all required attrs were seen.
  for (const auto& attr_it : attrs) {
    if (attr_it.second.required &&
        seen_attrs.find(attr_it.first) == seen_attrs.end()) {
      return Error(loc, StrFormat("attribute %s is expected but not seen",
                                  attr_it.first));
    }
  }
  return true;
}

bool HloParserImpl::ParseAttributeHelper(
    const absl::flat_hash_map<std::string, AttrConfig>& attrs,
    absl::flat_hash_set<std::string>* seen_attrs) {
  LocTy loc = lexer_.GetLoc();
  std::string name;
  if (!ParseAttributeName(&name)) {
    return Error(loc, "error parsing attributes");
  }
  VLOG(3) << "Parsing attribute " << name;
  if (!seen_attrs->insert(name).second) {
    return Error(loc, StrFormat("attribute %s already exists", name));
  }
  auto attr_it = attrs.find(name);
  if (attr_it == attrs.end()) {
    std::string allowed_attrs;
    if (attrs.empty()) {
      allowed_attrs = "No attributes are allowed here.";
    } else {
      allowed_attrs =
          StrCat("Allowed attributes: ",
                 StrJoin(attrs, ", ",
                         [&](std::string* out,
                             const std::pair<std::string, AttrConfig>& kv) {
                           StrAppend(out, kv.first);
                         }));
    }
    return Error(loc, StrFormat("unexpected attribute \"%s\".  %s", name,
                                allowed_attrs));
  }
  AttrTy attr_type = attr_it->second.attr_type;
  void* attr_out_ptr = attr_it->second.result;
  bool success = [&] {
    LocTy attr_loc = lexer_.GetLoc();
    switch (attr_type) {
      case AttrTy::kBool: {
        bool result;
        if (!ParseBool(&result)) {
          return false;
        }
        static_cast<optional<bool>*>(attr_out_ptr)->emplace(result);
        return true;
      }
      case AttrTy::kInt64: {
        int64 result;
        if (!ParseInt64(&result)) {
          return false;
        }
        static_cast<optional<int64>*>(attr_out_ptr)->emplace(result);
        return true;
      }
      case AttrTy::kInt32: {
        int64 result;
        if (!ParseInt64(&result)) {
          return false;
        }
        if (result != static_cast<int32>(result)) {
          return Error(attr_loc, "value out of range for int32");
        }
        static_cast<optional<int32>*>(attr_out_ptr)
            ->emplace(static_cast<int32>(result));
        return true;
      }
      case AttrTy::kFloat: {
        double result;
        if (!ParseDouble(&result)) {
          return false;
        }
        if (result > std::numeric_limits<float>::max() ||
            result < std::numeric_limits<float>::lowest()) {
          return Error(attr_loc, "value out of range for float");
        }
        static_cast<optional<float>*>(attr_out_ptr)
            ->emplace(static_cast<float>(result));
        return true;
      }
      case AttrTy::kHloComputation: {
        HloComputation* result = nullptr;
        if (!ParseHloComputation(&result)) {
          return false;
        }
        static_cast<optional<HloComputation*>*>(attr_out_ptr)->emplace(result);
        return true;
      }
      case AttrTy::kBracedHloComputationList: {
        std::vector<HloComputation*> result;
        if (!ParseHloComputationList(&result)) {
          return false;
        }
        static_cast<optional<std::vector<HloComputation*>>*>(attr_out_ptr)
            ->emplace(result);
        return true;
      }
      case AttrTy::kFftType: {
        FftType result;
        if (!ParseFftType(&result)) {
          return false;
        }
        static_cast<optional<FftType>*>(attr_out_ptr)->emplace(result);
        return true;
      }
      case AttrTy::kComparisonDirection: {
        ComparisonDirection result;
        if (!ParseComparisonDirection(&result)) {
          return false;
        }
        static_cast<optional<ComparisonDirection>*>(attr_out_ptr)
            ->emplace(result);
        return true;
      }
      case AttrTy::kEnum: {
        if (lexer_.GetKind() != TokKind::kIdent) {
          return TokenError("expects an enumeration value");
        }
        std::string result = lexer_.GetStrVal();
        lexer_.Lex();
        static_cast<optional<std::string>*>(attr_out_ptr)->emplace(result);
        return true;
      }
      case AttrTy::kWindow: {
        Window result;
        if (!ParseWindow(&result, /*expect_outer_curlies=*/true)) {
          return false;
        }
        static_cast<optional<Window>*>(attr_out_ptr)->emplace(result);
        return true;
      }
      case AttrTy::kConvolutionDimensionNumbers: {
        ConvolutionDimensionNumbers result;
        if (!ParseConvolutionDimensionNumbers(&result)) {
          return false;
        }
        static_cast<optional<ConvolutionDimensionNumbers>*>(attr_out_ptr)
            ->emplace(result);
        return true;
      }
      case AttrTy::kSharding: {
        OpSharding sharding;
        if (!ParseSharding(&sharding)) {
          return false;
        }
        static_cast<optional<OpSharding>*>(attr_out_ptr)->emplace(sharding);
        return true;
      }
      case AttrTy::kFrontendAttributes: {
        FrontendAttributes frontend_attributes;
        if (!ParseFrontendAttributes(&frontend_attributes)) {
          return false;
        }
        static_cast<optional<FrontendAttributes>*>(attr_out_ptr)
            ->emplace(frontend_attributes);
        return true;
      }
      case AttrTy::kParameterReplication: {
        ParameterReplication parameter_replication;
        if (!ParseParameterReplication(&parameter_replication)) {
          return false;
        }
        static_cast<optional<ParameterReplication>*>(attr_out_ptr)
            ->emplace(parameter_replication);
        return true;
      }
      case AttrTy::kInstructionList: {
        std::vector<HloInstruction*> result;
        if (!ParseInstructionNames(&result)) {
          return false;
        }
        static_cast<optional<std::vector<HloInstruction*>>*>(attr_out_ptr)
            ->emplace(result);
        return true;
      }
      case AttrTy::kFusionKind: {
        HloInstruction::FusionKind result;
        if (!ParseFusionKind(&result)) {
          return false;
        }
        static_cast<optional<HloInstruction::FusionKind>*>(attr_out_ptr)
            ->emplace(result);
        return true;
      }
      case AttrTy::kBracedInt64List: {
        std::vector<int64> result;
        if (!ParseInt64List(TokKind::kLbrace, TokKind::kRbrace, TokKind::kComma,
                            &result)) {
          return false;
        }
        static_cast<optional<std::vector<int64>>*>(attr_out_ptr)
            ->emplace(result);
        return true;
      }
      case AttrTy::kBracedInt64ListList: {
        std::vector<std::vector<int64>> result;
        if (!ParseInt64ListList(TokKind::kLbrace, TokKind::kRbrace,
                                TokKind::kComma, &result)) {
          return false;
        }
        static_cast<optional<std::vector<std::vector<int64>>>*>(attr_out_ptr)
            ->emplace(result);
        return true;
      }
      case AttrTy::kSliceRanges: {
        SliceRanges result;
        if (!ParseSliceRanges(&result)) {
          return false;
        }
        static_cast<optional<SliceRanges>*>(attr_out_ptr)->emplace(result);
        return true;
      }
      case AttrTy::kPaddingConfig: {
        PaddingConfig result;
        if (!ParsePaddingConfig(&result)) {
          return false;
        }
        static_cast<optional<PaddingConfig>*>(attr_out_ptr)->emplace(result);
        return true;
      }
      case AttrTy::kString: {
        std::string result;
        if (!ParseString(&result)) {
          return false;
        }
        static_cast<optional<std::string>*>(attr_out_ptr)->emplace(result);
        return true;
      }
      case AttrTy::kMetadata: {
        OpMetadata result;
        if (!ParseMetadata(&result)) {
          return false;
        }
        static_cast<optional<OpMetadata>*>(attr_out_ptr)->emplace(result);
        return true;
      }
      case AttrTy::kDistribution: {
        RandomDistribution result;
        if (!ParseRandomDistribution(&result)) {
          return false;
        }
        static_cast<optional<RandomDistribution>*>(attr_out_ptr)
            ->emplace(result);
        return true;
      }
      case AttrTy::kDomain: {
        return ParseDomain(static_cast<DomainData*>(attr_out_ptr));
      }
      case AttrTy::kPrecisionList: {
        std::vector<PrecisionConfig::Precision> result;
        if (!ParsePrecisionList(&result)) {
          return false;
        }
        static_cast<optional<std::vector<PrecisionConfig::Precision>>*>(
            attr_out_ptr)
            ->emplace(result);
        return true;
      }
      case AttrTy::kShapeList: {
        std::vector<Shape> result;
        if (!ParseShapeList(&result)) {
          return false;
        }
        static_cast<optional<std::vector<Shape>>*>(attr_out_ptr)
            ->emplace(result);
        return true;
      }
      case AttrTy::kRandomAlgorithm: {
        RandomAlgorithm result;
        if (!ParseRandomAlgorithm(&result)) {
          return false;
        }
        static_cast<optional<RandomAlgorithm>*>(attr_out_ptr)->emplace(result);
        return true;
      }
      case AttrTy::kAliasing: {
        AliasingData aliasing_data;
        if (!ParseAliasing(&aliasing_data)) {
          return false;
        }
        static_cast<optional<AliasingData>*>(attr_out_ptr)
            ->emplace(aliasing_data);
        return true;
      }
    }
  }();
  if (!success) {
    return Error(loc, StrFormat("error parsing attribute %s", name));
  }
  return true;
}

bool HloParserImpl::CopyAttributeToProtoMessage(
    absl::flat_hash_set<std::string> non_proto_attrs,
    const absl::flat_hash_map<std::string, AttrConfig>& attrs,
    tensorflow::protobuf::Message* message) {
  const tensorflow::protobuf::Descriptor* descriptor = message->GetDescriptor();
  const tensorflow::protobuf::Reflection* reflection = message->GetReflection();

  for (const auto& p : attrs) {
    const std::string& name = p.first;
    if (non_proto_attrs.find(name) != non_proto_attrs.end()) {
      continue;
    }
    const tensorflow::protobuf::FieldDescriptor* fd =
        descriptor->FindFieldByName(name);
    if (!fd) {
      std::string allowed_attrs = "Allowed attributes: ";

      for (int i = 0; i < descriptor->field_count(); ++i) {
        if (i == 0) {
          absl::StrAppend(&allowed_attrs, descriptor->field(i)->name());
        } else {
          absl::StrAppend(&allowed_attrs, ", ", descriptor->field(i)->name());
        }
      }
      return TokenError(
          StrFormat("unexpected attribute \"%s\".  %s", name, allowed_attrs));
    }

    CHECK(!fd->is_repeated());  // Repeated fields not implemented.
    bool success = [&] {
      switch (fd->type()) {
        case tensorflow::protobuf::FieldDescriptor::TYPE_BOOL: {
          auto attr_value = static_cast<optional<bool>*>(p.second.result);
          if (attr_value->has_value()) {
            reflection->SetBool(message, fd, **attr_value);
          }
          return true;
        }
        case tensorflow::protobuf::FieldDescriptor::TYPE_ENUM: {
          auto attr_value =
              static_cast<optional<std::string>*>(p.second.result);
          if (attr_value->has_value()) {
            const tensorflow::protobuf::EnumValueDescriptor* evd =
                fd->enum_type()->FindValueByName(**attr_value);
            reflection->SetEnum(message, fd, evd);
          }
          return true;
        }
        default:
          return false;
      }
    }();

    if (!success) {
      return TokenError(StrFormat("error parsing attribute %s", name));
    }
  }

  return true;
}

// attributes ::= (',' attribute)*
bool HloParserImpl::ParseAttributesAsProtoMessage(
    const absl::flat_hash_map<std::string, AttrConfig>& non_proto_attrs,
    tensorflow::protobuf::Message* message) {
  const tensorflow::protobuf::Descriptor* descriptor = message->GetDescriptor();
  absl::flat_hash_map<std::string, AttrConfig> attrs;

  // Storage for attributes.
  std::vector<optional<bool>> bool_params;
  std::vector<optional<std::string>> string_params;
  // Reserve enough capacity to make sure that the vector is not growing, so we
  // can rely on the pointers to stay valid.
  bool_params.reserve(descriptor->field_count());
  string_params.reserve(descriptor->field_count());

  // Populate the storage of expected attributes from the protobuf description.
  for (int field_idx = 0; field_idx < descriptor->field_count(); field_idx++) {
    const tensorflow::protobuf::FieldDescriptor* fd =
        descriptor->field(field_idx);
    const std::string& field_name = fd->name();
    switch (fd->type()) {
      case tensorflow::protobuf::FieldDescriptor::TYPE_BOOL: {
        bool_params.emplace_back(absl::nullopt);
        attrs[field_name] = {/*is_required*/ false, AttrTy::kBool,
                             &bool_params.back()};
        break;
      }
      case tensorflow::protobuf::FieldDescriptor::TYPE_ENUM: {
        string_params.emplace_back(absl::nullopt);
        attrs[field_name] = {/*is_required*/ false, AttrTy::kEnum,
                             &string_params.back()};
        break;
      }
      default:
        return TokenError(absl::StrFormat(
            "Unexpected protocol buffer type: %s ", fd->DebugString()));
    }
  }

  absl::flat_hash_set<std::string> non_proto_attrs_names;
  non_proto_attrs_names.reserve(non_proto_attrs.size());
  for (const auto& p : non_proto_attrs) {
    const std::string& attr_name = p.first;
    // If an attribute is both specified within 'non_proto_attrs' and an
    // attribute of the proto message, we prefer the attribute of the proto
    // message.
    if (attrs.find(attr_name) == attrs.end()) {
      non_proto_attrs_names.insert(attr_name);
      attrs[attr_name] = p.second;
    }
  }

  if (!ParseAttributes(attrs)) {
    return false;
  }

  return CopyAttributeToProtoMessage(non_proto_attrs_names, attrs, message);
}

bool HloParserImpl::ParseComputationName(HloComputation** value) {
  std::string name;
  LocTy loc = lexer_.GetLoc();
  if (!ParseName(&name)) {
    return Error(loc, "expects computation name");
  }
  std::pair<HloComputation*, LocTy>* computation =
      tensorflow::gtl::FindOrNull(computation_pool_, name);
  if (computation == nullptr) {
    return Error(loc, StrCat("computation does not exist: ", name));
  }
  *value = computation->first;
  return true;
}

// ::= '{' size stride? pad? lhs_dilate? rhs_dilate? '}'
// The subattributes can appear in any order. 'size=' is required, others are
// optional.
bool HloParserImpl::ParseWindow(Window* window, bool expect_outer_curlies) {
  LocTy loc = lexer_.GetLoc();
  if (expect_outer_curlies &&
      !ParseToken(TokKind::kLbrace, "expected '{' to start window attribute")) {
    return false;
  }

  std::vector<int64> size;
  std::vector<int64> stride;
  std::vector<std::vector<int64>> pad;
  std::vector<int64> lhs_dilate;
  std::vector<int64> rhs_dilate;
  std::vector<int64> rhs_reversal;
  const auto end_token =
      expect_outer_curlies ? TokKind::kRbrace : TokKind::kEof;
  while (lexer_.GetKind() != end_token) {
    LocTy attr_loc = lexer_.GetLoc();
    std::string field_name;
    if (!ParseAttributeName(&field_name)) {
      return Error(attr_loc, "expects sub-attributes in window");
    }
    bool ok = [&] {
      if (field_name == "size") {
        return ParseDxD("size", &size);
      }
      if (field_name == "stride") {
        return ParseDxD("stride", &stride);
      }
      if (field_name == "lhs_dilate") {
        return ParseDxD("lhs_dilate", &lhs_dilate);
      }
      if (field_name == "rhs_dilate") {
        return ParseDxD("rls_dilate", &rhs_dilate);
      }
      if (field_name == "pad") {
        return ParseWindowPad(&pad);
      }
      if (field_name == "rhs_reversal") {
        return ParseDxD("rhs_reversal", &rhs_reversal);
      }
      return Error(attr_loc, StrCat("unexpected attribute name: ", field_name));
    }();
    if (!ok) {
      return false;
    }
  }

  if (!stride.empty() && stride.size() != size.size()) {
    return Error(loc, "expects 'stride=' has the same size as 'size='");
  }
  if (!lhs_dilate.empty() && lhs_dilate.size() != size.size()) {
    return Error(loc, "expects 'lhs_dilate=' has the same size as 'size='");
  }
  if (!rhs_dilate.empty() && rhs_dilate.size() != size.size()) {
    return Error(loc, "expects 'rhs_dilate=' has the same size as 'size='");
  }
  if (!pad.empty() && pad.size() != size.size()) {
    return Error(loc, "expects 'pad=' has the same size as 'size='");
  }

  for (int i = 0; i < size.size(); i++) {
    window->add_dimensions()->set_size(size[i]);
    if (!pad.empty()) {
      window->mutable_dimensions(i)->set_padding_low(pad[i][0]);
      window->mutable_dimensions(i)->set_padding_high(pad[i][1]);
    }
    // If some field is not present, it has the default value.
    window->mutable_dimensions(i)->set_stride(stride.empty() ? 1 : stride[i]);
    window->mutable_dimensions(i)->set_base_dilation(
        lhs_dilate.empty() ? 1 : lhs_dilate[i]);
    window->mutable_dimensions(i)->set_window_dilation(
        rhs_dilate.empty() ? 1 : rhs_dilate[i]);
    window->mutable_dimensions(i)->set_window_reversal(
        rhs_reversal.empty() ? false : (rhs_reversal[i] == 1));
  }
  return !expect_outer_curlies ||
         ParseToken(TokKind::kRbrace, "expected '}' to end window attribute");
}

// This is the inverse of HloInstruction::ConvolutionDimensionNumbersToString.
// Thestring looks like "dim_labels=0bf_0io->0bf".
bool HloParserImpl::ParseConvolutionDimensionNumbers(
    ConvolutionDimensionNumbers* dnums) {
  if (lexer_.GetKind() != TokKind::kDimLabels) {
    return TokenError("expects dim labels pattern, e.g., 'bf0_0io->0bf'");
  }
  std::string str = lexer_.GetStrVal();

  // The str is expected to have 3 items, lhs, rhs, out, and it must look like
  // lhs_rhs->out, that is, the first separator is "_" and the second is "->".
  std::vector<std::string> split1 = absl::StrSplit(str, '_');
  if (split1.size() != 2) {
    LOG(FATAL) << "expects 3 items: lhs, rhs, and output dims, but sees "
               << str;
  }
  std::vector<std::string> split2 = absl::StrSplit(split1[1], "->");
  if (split2.size() != 2) {
    LOG(FATAL) << "expects 3 items: lhs, rhs, and output dims, but sees "
               << str;
  }
  absl::string_view lhs = split1[0];
  absl::string_view rhs = split2[0];
  absl::string_view out = split2[1];

  const int64 rank = lhs.length();
  if (rank != rhs.length() || rank != out.length()) {
    return TokenError(
        "convolution lhs, rhs, and output must have the same rank");
  }
  if (rank < 2) {
    return TokenError("convolution rank must >=2");
  }

  auto is_unique = [](std::string str) -> bool {
    absl::c_sort(str);
    return std::unique(str.begin(), str.end()) == str.end();
  };

  // lhs
  {
    if (!is_unique(std::string(lhs))) {
      return TokenError(
          StrCat("expects unique lhs dimension numbers, but sees ", lhs));
    }
    for (int i = 0; i < rank - 2; i++) {
      dnums->add_input_spatial_dimensions(-1);
    }
    for (int i = 0; i < rank; i++) {
      char c = lhs[i];
      if (c == 'b') {
        dnums->set_input_batch_dimension(i);
      } else if (c == 'f') {
        dnums->set_input_feature_dimension(i);
      } else if (c < '0' + rank && c >= '0') {
        dnums->set_input_spatial_dimensions(c - '0', i);
      } else {
        return TokenError(
            StrFormat("expects [0-%dbf] in lhs dimension numbers", rank - 1));
      }
    }
  }
  // rhs
  {
    if (!is_unique(std::string(rhs))) {
      return TokenError(
          StrCat("expects unique rhs dimension numbers, but sees ", rhs));
    }
    for (int i = 0; i < rank - 2; i++) {
      dnums->add_kernel_spatial_dimensions(-1);
    }
    for (int i = 0; i < rank; i++) {
      char c = rhs[i];
      if (c == 'i') {
        dnums->set_kernel_input_feature_dimension(i);
      } else if (c == 'o') {
        dnums->set_kernel_output_feature_dimension(i);
      } else if (c < '0' + rank && c >= '0') {
        dnums->set_kernel_spatial_dimensions(c - '0', i);
      } else {
        return TokenError(
            StrFormat("expects [0-%dio] in rhs dimension numbers", rank - 1));
      }
    }
  }
  // output
  {
    if (!is_unique(std::string(out))) {
      return TokenError(
          StrCat("expects unique output dimension numbers, but sees ", out));
    }
    for (int i = 0; i < rank - 2; i++) {
      dnums->add_output_spatial_dimensions(-1);
    }
    for (int i = 0; i < rank; i++) {
      char c = out[i];
      if (c == 'b') {
        dnums->set_output_batch_dimension(i);
      } else if (c == 'f') {
        dnums->set_output_feature_dimension(i);
      } else if (c < '0' + rank && c >= '0') {
        dnums->set_output_spatial_dimensions(c - '0', i);
      } else {
        return TokenError(StrFormat(
            "expects [0-%dbf] in output dimension numbers", rank - 1));
      }
    }
  }

  lexer_.Lex();
  return true;
}

// ::= '{' ranges '}'
//   ::= /*empty*/
//   ::= range (',' range)*
// range ::= '[' start ':' limit (':' stride)? ']'
//
// The slice ranges are printed as:
//
//  {[dim0_start:dim0_limit:dim0stride], [dim1_start:dim1_limit], ...}
//
// This function extracts the starts, limits, and strides as 3 vectors to the
// result. If stride is not present, stride is 1. For example, if the slice
// ranges is printed as:
//
//  {[2:3:4], [5:6:7], [8:9]}
//
// The parsed result will be:
//
//  {/*starts=*/{2, 5, 8}, /*limits=*/{3, 6, 9}, /*strides=*/{4, 7, 1}}
//
bool HloParserImpl::ParseSliceRanges(SliceRanges* result) {
  if (!ParseToken(TokKind::kLbrace, "expects '{' to start ranges")) {
    return false;
  }
  std::vector<std::vector<int64>> ranges;
  if (lexer_.GetKind() == TokKind::kRbrace) {
    // empty
    return ParseToken(TokKind::kRbrace, "expects '}' to end ranges");
  }
  do {
    LocTy loc = lexer_.GetLoc();
    ranges.emplace_back();
    if (!ParseInt64List(TokKind::kLsquare, TokKind::kRsquare, TokKind::kColon,
                        &ranges.back())) {
      return false;
    }
    const auto& range = ranges.back();
    if (range.size() != 2 && range.size() != 3) {
      return Error(loc,
                   StrFormat("expects [start:limit:step] or [start:limit], "
                             "but sees %d elements.",
                             range.size()));
    }
  } while (EatIfPresent(TokKind::kComma));

  for (const auto& range : ranges) {
    result->starts.push_back(range[0]);
    result->limits.push_back(range[1]);
    result->strides.push_back(range.size() == 3 ? range[2] : 1);
  }
  return ParseToken(TokKind::kRbrace, "expects '}' to end ranges");
}

// precisionlist ::= start precision_elements end
// precision_elements
//   ::= /*empty*/
//   ::= precision_val (delim precision_val)*
bool HloParserImpl::ParsePrecisionList(
    std::vector<PrecisionConfig::Precision>* result) {
  auto parse_and_add_item = [&]() {
    PrecisionConfig::Precision item;
    if (!ParsePrecision(&item)) {
      return false;
    }
    result->push_back(item);
    return true;
  };
  return ParseList(TokKind::kLbrace, TokKind::kRbrace, TokKind::kComma,
                   parse_and_add_item);
}

bool HloParserImpl::ParseHloComputation(HloComputation** result) {
  if (lexer_.GetKind() == TokKind::kLbrace) {
    // This means it is a nested computation.
    return ParseInstructionList(result, /*computation_name=*/"_");
  }
  // This means it is a computation name.
  return ParseComputationName(result);
}

bool HloParserImpl::ParseHloComputationList(
    std::vector<HloComputation*>* result) {
  auto parse_and_add_item = [&]() {
    HloComputation* computation;
    if (!ParseHloComputation(&computation)) {
      return false;
    }
    LOG(INFO) << "parsed computation " << computation->name();
    result->push_back(computation);
    return true;
  };
  return ParseList(TokKind::kLbrace, TokKind::kRbrace, TokKind::kComma,
                   parse_and_add_item);
}

// shapelist ::= '{' shapes '}'
// precision_elements
//   ::= /*empty*/
//   ::= shape (',' shape)*
bool HloParserImpl::ParseShapeList(std::vector<Shape>* result) {
  auto parse_and_add_item = [&]() {
    Shape shape;
    if (!ParseShape(&shape)) {
      return false;
    }
    result->push_back(std::move(shape));
    return true;
  };
  return ParseList(TokKind::kLbrace, TokKind::kRbrace, TokKind::kComma,
                   parse_and_add_item);
}

// int64list ::= start int64_elements end
// int64_elements
//   ::= /*empty*/
//   ::= int64_val (delim int64_val)*
bool HloParserImpl::ParseInt64List(const TokKind start, const TokKind end,
                                   const TokKind delim,
                                   std::vector<int64>* result) {
  auto parse_and_add_item = [&]() {
    int64 i;
    if (!ParseInt64(&i)) {
      return false;
    }
    result->push_back(i);
    return true;
  };
  return ParseList(start, end, delim, parse_and_add_item);
}

// int64listlist ::= start int64list_elements end
// int64list_elements
//   ::= /*empty*/
//   ::= int64list (delim int64list)*
// int64list ::= start int64_elements end
// int64_elements
//   ::= /*empty*/
//   ::= int64_val (delim int64_val)*
bool HloParserImpl::ParseInt64ListList(
    const TokKind start, const TokKind end, const TokKind delim,
    std::vector<std::vector<int64>>* result) {
  auto parse_and_add_item = [&]() {
    std::vector<int64> item;
    if (!ParseInt64List(start, end, delim, &item)) {
      return false;
    }
    result->push_back(item);
    return true;
  };
  return ParseList(start, end, delim, parse_and_add_item);
}

bool HloParserImpl::ParseList(const TokKind start, const TokKind end,
                              const TokKind delim,
                              const std::function<bool()>& parse_and_add_item) {
  if (!ParseToken(start, StrCat("expects a list starting with ",
                                TokKindToString(start)))) {
    return false;
  }
  if (lexer_.GetKind() == end) {
    // empty
  } else {
    do {
      if (!parse_and_add_item()) {
        return false;
      }
    } while (EatIfPresent(delim));
  }
  return ParseToken(
      end, StrCat("expects a list to end with ", TokKindToString(end)));
}

// param_list_to_shape ::= param_list '->' shape
bool HloParserImpl::ParseParamListToShape(Shape* shape, LocTy* shape_loc) {
  if (!ParseParamList() || !ParseToken(TokKind::kArrow, "expects '->'")) {
    return false;
  }
  *shape_loc = lexer_.GetLoc();
  return ParseShape(shape);
}

bool HloParserImpl::CanBeParamListToShape() {
  return lexer_.GetKind() == TokKind::kLparen;
}

// param_list ::= '(' param_list1 ')'
// param_list1
//   ::= /*empty*/
//   ::= param (',' param)*
// param ::= name shape
bool HloParserImpl::ParseParamList() {
  if (!ParseToken(TokKind::kLparen,
                  "expects '(' at the beginning of param list")) {
    return false;
  }

  if (lexer_.GetKind() == TokKind::kRparen) {
    // empty
  } else {
    do {
      Shape shape;
      std::string name;
      if (!ParseName(&name) || !ParseShape(&shape)) {
        return false;
      }
    } while (EatIfPresent(TokKind::kComma));
  }
  return ParseToken(TokKind::kRparen, "expects ')' at the end of param list");
}

// dimension_sizes ::= '[' dimension_list ']'
// dimension_list
//   ::= /*empty*/
//   ::= <=? int64 (',' param)*
// param ::= name shape
bool HloParserImpl::ParseDimensionSizes(std::vector<int64>* dimension_sizes,
                                        std::vector<bool>* dynamic_dimensions) {
  auto parse_and_add_item = [&]() {
    int64 i;
    bool is_dynamic = false;
    if (lexer_.GetKind() == TokKind::kLeq) {
      is_dynamic = true;
      lexer_.Lex();
    }
    if (!ParseInt64(&i)) {
      return false;
    }
    dimension_sizes->push_back(i);
    dynamic_dimensions->push_back(is_dynamic);
    return true;
  };
  return ParseList(TokKind::kLsquare, TokKind::kRsquare, TokKind::kComma,
                   parse_and_add_item);
}

// tiles
//   ::= /*empty*/
//   ::= 'T' '(' dim_list ')'
// dim_list
//   ::= /*empty*/
//   ::= (int64 | '*') (',' (int64 | '*'))*
bool HloParserImpl::ParseTiles(std::vector<Tile>* tiles) {
  auto parse_and_add_tile_dimension = [&]() {
    tensorflow::int64 i;
    if (ParseInt64(&i)) {
      tiles->back().add_dimensions(i);
      return true;
    }
    if (lexer_.GetKind() == TokKind::kAsterisk) {
      tiles->back().add_dimensions(Tile::kCombineDimension);
      lexer_.Lex();
      return true;
    }
    return false;
  };

  do {
    tiles->push_back(Tile());
    if (!ParseList(TokKind::kLparen, TokKind::kRparen, TokKind::kComma,
                   parse_and_add_tile_dimension)) {
      return false;
    }
  } while (lexer_.GetKind() == TokKind::kLparen);
  return true;
}

// int_attribute
//   ::= /*empty*/
//   ::= attr_token '(' attr_value ')'
// attr_token
//   ::= 'E' | 'S'
// attr_value
//   ::= int64
bool HloParserImpl::ParseLayoutIntAttribute(
    int64* attr_value, absl::string_view attr_description) {
  if (!ParseToken(TokKind::kLparen,
                  StrCat("expects ", attr_description, " to start with ",
                         TokKindToString(TokKind::kLparen)))) {
    return false;
  }
  if (!ParseInt64(attr_value)) {
    return false;
  }
  if (!ParseToken(TokKind::kRparen,
                  StrCat("expects ", attr_description, " to end with ",
                         TokKindToString(TokKind::kRparen)))) {
    return false;
  }
  return true;
}

// layout ::= '{' int64_list (':' tiles element_size_in_bits memory_space)? '}'
// element_size_in_bits
//   ::= /*empty*/
//   ::= 'E' '(' int64 ')'
// memory_space
//   ::= /*empty*/
//   ::= 'S' '(' int64 ')'
bool HloParserImpl::ParseLayout(Layout* layout) {
  std::vector<int64> minor_to_major;
  std::vector<Tile> tiles;
  tensorflow::int64 element_size_in_bits = 0;
  tensorflow::int64 memory_space = 0;

  auto parse_and_add_item = [&]() {
    int64 i;
    if (!ParseInt64(&i)) {
      return false;
    }
    minor_to_major.push_back(i);
    return true;
  };

  if (!ParseToken(TokKind::kLbrace,
                  StrCat("expects layout to start with ",
                         TokKindToString(TokKind::kLbrace)))) {
    return false;
  }
  if (lexer_.GetKind() != TokKind::kRbrace) {
    if (lexer_.GetKind() == TokKind::kInt) {
      // Parse minor to major.
      do {
        if (!parse_and_add_item()) {
          return false;
        }
      } while (EatIfPresent(TokKind::kComma));
    }

    if (lexer_.GetKind() == TokKind::kColon) {
      lexer_.Lex();
      if (lexer_.GetKind() == TokKind::kIdent && lexer_.GetStrVal() == "T") {
        lexer_.Lex();
        ParseTiles(&tiles);
      }

      if (lexer_.GetKind() == TokKind::kIdent && lexer_.GetStrVal() == "E") {
        lexer_.Lex();
        ParseLayoutIntAttribute(&element_size_in_bits, "element size in bits");
      }

      if (lexer_.GetKind() == TokKind::kIdent && lexer_.GetStrVal() == "S") {
        lexer_.Lex();
        ParseLayoutIntAttribute(&memory_space, "memory space");
      }
    }
  }
  if (!ParseToken(TokKind::kRbrace,
                  StrCat("expects layout to end with ",
                         TokKindToString(TokKind::kRbrace)))) {
    return false;
  }

  std::vector<Tile> vec_tiles(tiles.size());
  for (int i = 0; i < tiles.size(); i++) {
    vec_tiles[i] = Tile(tiles[i]);
  }
  *layout = LayoutUtil::MakeLayout(minor_to_major, vec_tiles,
                                   element_size_in_bits, memory_space);
  return true;
}

// shape ::= shape_val_
// shape ::= '(' tuple_elements ')'
// tuple_elements
//   ::= /*empty*/
//   ::= shape (',' shape)*
bool HloParserImpl::ParseShape(Shape* result) {
  if (EatIfPresent(TokKind::kLparen)) {  // Tuple
    std::vector<Shape> shapes;
    if (lexer_.GetKind() == TokKind::kRparen) {
      /*empty*/
    } else {
      // shape (',' shape)*
      do {
        shapes.emplace_back();
        if (!ParseShape(&shapes.back())) {
          return false;
        }
      } while (EatIfPresent(TokKind::kComma));
    }
    *result = ShapeUtil::MakeTupleShape(shapes);
    return ParseToken(TokKind::kRparen, "expects ')' at the end of tuple.");
  }

  if (lexer_.GetKind() != TokKind::kPrimitiveType) {
    return TokenError(absl::StrCat("expected primitive type, saw ",
                                   TokKindToString(lexer_.GetKind())));
  }
  PrimitiveType primitive_type = lexer_.GetPrimitiveTypeVal();
  lexer_.Lex();

  // Each element contains a dimension size and a bool indicating whether this
  // is a dynamic dimension.
  std::vector<int64> dimension_sizes;
  std::vector<bool> dynamic_dimensions;
  if (!ParseDimensionSizes(&dimension_sizes, &dynamic_dimensions)) {
    return false;
  }
  result->set_element_type(primitive_type);
  for (int i = 0; i < dimension_sizes.size(); ++i) {
    result->add_dimensions(dimension_sizes[i]);
    result->set_dynamic_dimension(i, dynamic_dimensions[i]);
  }
  LayoutUtil::SetToDefaultLayout(result);

  // We need to lookahead to see if a following open brace is the start of a
  // layout. The specific problematic case is:
  //
  // ENTRY %foo (x: f32[42]) -> f32[123] {
  //  ...
  // }
  //
  // The open brace could either be the start of a computation or the start of a
  // layout for the f32[123] shape. We consider it the start of a layout if the
  // next token after the open brace is an integer or a colon.
  if (lexer_.GetKind() == TokKind::kLbrace &&
      (lexer_.LookAhead() == TokKind::kInt ||
       lexer_.LookAhead() == TokKind::kColon)) {
    Layout layout;
    if (!ParseLayout(&layout)) {
      return false;
    }
    if (layout.minor_to_major_size() != result->rank()) {
      return Error(
          lexer_.GetLoc(),
          StrFormat("Dimensions size is %ld, but minor to major size is %ld.",
                    result->rank(), layout.minor_to_major_size()));
    }
    *result->mutable_layout() = layout;
  }
  return true;
}

bool HloParserImpl::CanBeShape() {
  // A non-tuple shape starts with a kPrimitiveType token; a tuple shape starts
  // with '('.
  return lexer_.GetKind() == TokKind::kPrimitiveType ||
         lexer_.GetKind() == TokKind::kLparen;
}

bool HloParserImpl::ParseName(std::string* result) {
  VLOG(3) << "ParseName";
  if (lexer_.GetKind() != TokKind::kIdent &&
      lexer_.GetKind() != TokKind::kName) {
    return TokenError("expects name");
  }
  *result = lexer_.GetStrVal();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseAttributeName(std::string* result) {
  if (lexer_.GetKind() != TokKind::kAttributeName) {
    return TokenError("expects attribute name");
  }
  *result = lexer_.GetStrVal();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseString(std::string* result) {
  VLOG(3) << "ParseString";
  if (lexer_.GetKind() != TokKind::kString) {
    return TokenError("expects string");
  }
  *result = lexer_.GetStrVal();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseDxD(const std::string& name,
                             std::vector<int64>* result) {
  LocTy loc = lexer_.GetLoc();
  if (!result->empty()) {
    return Error(loc, StrFormat("sub-attribute '%s=' already exists", name));
  }
  // 1D
  if (lexer_.GetKind() == TokKind::kInt) {
    int64 number;
    if (!ParseInt64(&number)) {
      return Error(loc, StrFormat("expects sub-attribute '%s=i'", name));
    }
    result->push_back(number);
    return true;
  }
  // 2D or higher.
  if (lexer_.GetKind() == TokKind::kDxD) {
    std::string str = lexer_.GetStrVal();
    if (!SplitToInt64s(str, 'x', result)) {
      return Error(loc, StrFormat("expects sub-attribute '%s=ixj...'", name));
    }
    lexer_.Lex();
    return true;
  }
  return TokenError("expects token type kInt or kDxD");
}

bool HloParserImpl::ParseWindowPad(std::vector<std::vector<int64>>* pad) {
  LocTy loc = lexer_.GetLoc();
  if (!pad->empty()) {
    return Error(loc, "sub-attribute 'pad=' already exists");
  }
  if (lexer_.GetKind() != TokKind::kPad) {
    return TokenError("expects window pad pattern, e.g., '0_0x3_3'");
  }
  std::string str = lexer_.GetStrVal();
  for (const auto& padding_dim_str : absl::StrSplit(str, 'x')) {
    std::vector<int64> low_high;
    if (!SplitToInt64s(padding_dim_str, '_', &low_high) ||
        low_high.size() != 2) {
      return Error(loc,
                   "expects padding_low and padding_high separated by '_'");
    }
    pad->push_back(low_high);
  }
  lexer_.Lex();
  return true;
}

// This is the inverse xla::ToString(PaddingConfig). The padding config string
// looks like "0_0_0x3_3_1". The string is first separated by 'x', each
// substring represents one PaddingConfigDimension. The substring is 3 (or 2)
// numbers joined by '_'.
bool HloParserImpl::ParsePaddingConfig(PaddingConfig* padding) {
  if (lexer_.GetKind() != TokKind::kPad) {
    return TokenError("expects padding config, e.g., '0_0_0x3_3_1'");
  }
  LocTy loc = lexer_.GetLoc();
  std::string str = lexer_.GetStrVal();
  for (const auto& padding_dim_str : absl::StrSplit(str, 'x')) {
    std::vector<int64> padding_dim;
    if (!SplitToInt64s(padding_dim_str, '_', &padding_dim) ||
        (padding_dim.size() != 2 && padding_dim.size() != 3)) {
      return Error(loc,
                   "expects padding config pattern like 'low_high_interior' or "
                   "'low_high'");
    }
    auto* dim = padding->add_dimensions();
    dim->set_edge_padding_low(padding_dim[0]);
    dim->set_edge_padding_high(padding_dim[1]);
    dim->set_interior_padding(padding_dim.size() == 3 ? padding_dim[2] : 0);
  }
  lexer_.Lex();
  return true;
}

// '{' metadata_string '}'
bool HloParserImpl::ParseMetadata(OpMetadata* metadata) {
  absl::flat_hash_map<std::string, AttrConfig> attrs;
  optional<std::string> op_type;
  optional<std::string> op_name;
  optional<std::string> source_file;
  optional<int32> source_line;
  attrs["op_type"] = {/*required=*/false, AttrTy::kString, &op_type};
  attrs["op_name"] = {/*required=*/false, AttrTy::kString, &op_name};
  attrs["source_file"] = {/*required=*/false, AttrTy::kString, &source_file};
  attrs["source_line"] = {/*required=*/false, AttrTy::kInt32, &source_line};
  if (!ParseSubAttributes(attrs)) {
    return false;
  }
  if (op_type) {
    metadata->set_op_type(*op_type);
  }
  if (op_name) {
    metadata->set_op_name(*op_name);
  }
  if (source_file) {
    metadata->set_source_file(*source_file);
  }
  if (source_line) {
    metadata->set_source_line(*source_line);
  }
  return true;
}

bool HloParserImpl::ParseOpcode(HloOpcode* result) {
  VLOG(3) << "ParseOpcode";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects opcode");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToHloOpcode(val);
  if (!status_or_result.ok()) {
    return TokenError(StrFormat("expects opcode but sees: %s, error: %s", val,
                                status_or_result.status().error_message()));
  }
  *result = status_or_result.ValueOrDie();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseFftType(FftType* result) {
  VLOG(3) << "ParseFftType";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects fft type");
  }
  std::string val = lexer_.GetStrVal();
  if (!FftType_Parse(val, result) || !FftType_IsValid(*result)) {
    return TokenError(StrFormat("expects fft type but sees: %s", val));
  }
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseComparisonDirection(ComparisonDirection* result) {
  VLOG(1) << "ParseComparisonDirection";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects comparison direction");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToComparisonDirection(val);
  if (!status_or_result.ok()) {
    return TokenError(
        StrFormat("expects comparison direction but sees: %s", val));
  }
  *result = status_or_result.ValueOrDie();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseFusionKind(HloInstruction::FusionKind* result) {
  VLOG(3) << "ParseFusionKind";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects fusion kind");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToFusionKind(val);
  if (!status_or_result.ok()) {
    return TokenError(StrFormat("expects fusion kind but sees: %s, error: %s",
                                val,
                                status_or_result.status().error_message()));
  }
  *result = status_or_result.ValueOrDie();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseRandomDistribution(RandomDistribution* result) {
  VLOG(3) << "ParseRandomDistribution";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects random distribution");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToRandomDistribution(val);
  if (!status_or_result.ok()) {
    return TokenError(
        StrFormat("expects random distribution but sees: %s, error: %s", val,
                  status_or_result.status().error_message()));
  }
  *result = status_or_result.ValueOrDie();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseRandomAlgorithm(RandomAlgorithm* result) {
  VLOG(3) << "ParseRandomAlgorithm";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects random algorithm");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToRandomAlgorithm(val);
  if (!status_or_result.ok()) {
    return TokenError(
        StrFormat("expects random algorithm but sees: %s, error: %s", val,
                  status_or_result.status().error_message()));
  }
  *result = status_or_result.ValueOrDie();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParsePrecision(PrecisionConfig::Precision* result) {
  VLOG(3) << "ParsePrecision";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects random distribution");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToPrecision(val);
  if (!status_or_result.ok()) {
    return TokenError(StrFormat("expects precision but sees: %s, error: %s",
                                val,
                                status_or_result.status().error_message()));
  }
  *result = status_or_result.ValueOrDie();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseInt64(int64* result) {
  VLOG(3) << "ParseInt64";
  if (lexer_.GetKind() != TokKind::kInt) {
    return TokenError("expects integer");
  }
  *result = lexer_.GetInt64Val();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseDouble(double* result) {
  switch (lexer_.GetKind()) {
    case TokKind::kDecimal: {
      double val = lexer_.GetDecimalVal();
      // If GetDecimalVal returns +/-inf, that means that we overflowed
      // `double`.
      if (std::isinf(val)) {
        return TokenError(StrCat("Constant is out of range for double (+/-",
                                 std::numeric_limits<double>::max(),
                                 ") and so is unparsable."));
      }
      *result = val;
      break;
    }
    case TokKind::kInt:
      *result = static_cast<double>(lexer_.GetInt64Val());
      break;
    case TokKind::kw_nan:
      *result = std::numeric_limits<double>::quiet_NaN();
      break;
    case TokKind::kw_inf:
      *result = std::numeric_limits<double>::infinity();
      break;
    case TokKind::kNegInf:
      *result = -std::numeric_limits<double>::infinity();
      break;
    default:
      return TokenError("expects decimal or integer");
  }
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseComplex(std::complex<double>* result) {
  if (lexer_.GetKind() != TokKind::kLparen) {
    return TokenError("expects '(' before complex number");
  }
  lexer_.Lex();

  double real;
  LocTy loc = lexer_.GetLoc();
  if (!ParseDouble(&real)) {
    return Error(loc,
                 "expect floating-point value for real part of complex number");
  }

  if (lexer_.GetKind() != TokKind::kComma) {
    return TokenError(
        absl::StrFormat("expect comma after real part of complex literal"));
  }
  lexer_.Lex();

  double imag;
  loc = lexer_.GetLoc();
  if (!ParseDouble(&imag)) {
    return Error(
        loc,
        "expect floating-point value for imaginary part of complex number");
  }

  if (lexer_.GetKind() != TokKind::kRparen) {
    return TokenError(absl::StrFormat("expect ')' after complex number"));
  }

  *result = std::complex<double>(real, imag);
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseBool(bool* result) {
  if (lexer_.GetKind() != TokKind::kw_true &&
      lexer_.GetKind() != TokKind::kw_false) {
    return TokenError("expects true or false");
  }
  *result = lexer_.GetKind() == TokKind::kw_true;
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseToken(TokKind kind, const std::string& msg) {
  VLOG(3) << "ParseToken " << TokKindToString(kind) << " " << msg;
  if (lexer_.GetKind() != kind) {
    return TokenError(msg);
  }
  lexer_.Lex();
  return true;
}

bool HloParserImpl::EatIfPresent(TokKind kind) {
  if (lexer_.GetKind() != kind) {
    return false;
  }
  lexer_.Lex();
  return true;
}

bool HloParserImpl::AddInstruction(const std::string& name,
                                   HloInstruction* instruction,
                                   LocTy name_loc) {
  auto result = current_name_table().insert({name, {instruction, name_loc}});
  if (!result.second) {
    Error(name_loc, StrCat("instruction already exists: ", name));
    return Error(/*loc=*/result.first->second.second,
                 "instruction previously defined here");
  }
  return true;
}

bool HloParserImpl::AddComputation(const std::string& name,
                                   HloComputation* computation,
                                   LocTy name_loc) {
  auto result = computation_pool_.insert({name, {computation, name_loc}});
  if (!result.second) {
    Error(name_loc, StrCat("computation already exists: ", name));
    return Error(/*loc=*/result.first->second.second,
                 "computation previously defined here");
  }
  return true;
}

StatusOr<Shape> HloParserImpl::ParseShapeOnly() {
  lexer_.Lex();
  Shape shape;
  if (!ParseShape(&shape)) {
    return InvalidArgument("Syntax error:\n%s", GetError());
  }
  if (lexer_.GetKind() != TokKind::kEof) {
    return InvalidArgument("Syntax error:\nExtra content after shape");
  }
  return shape;
}

StatusOr<HloSharding> HloParserImpl::ParseShardingOnly() {
  lexer_.Lex();
  OpSharding op_sharding;
  if (!ParseSharding(&op_sharding)) {
    return InvalidArgument("Syntax error:\n%s", GetError());
  }
  if (lexer_.GetKind() != TokKind::kEof) {
    return InvalidArgument("Syntax error:\nExtra content after sharding");
  }
  return HloSharding::FromProto(op_sharding);
}

StatusOr<FrontendAttributes> HloParserImpl::ParseFrontendAttributesOnly() {
  lexer_.Lex();
  FrontendAttributes attributes;
  if (!ParseFrontendAttributes(&attributes)) {
    return InvalidArgument("Syntax error:\n%s", GetError());
  }
  if (lexer_.GetKind() != TokKind::kEof) {
    return InvalidArgument(
        "Syntax error:\nExtra content after frontend attributes");
  }
  return attributes;
}

StatusOr<std::vector<bool>> HloParserImpl::ParseParameterReplicationOnly() {
  lexer_.Lex();
  ParameterReplication parameter_replication;
  if (!ParseParameterReplication(&parameter_replication)) {
    return InvalidArgument("Syntax error:\n%s", GetError());
  }
  if (lexer_.GetKind() != TokKind::kEof) {
    return InvalidArgument(
        "Syntax error:\nExtra content after parameter replication");
  }
  return std::vector<bool>(
      parameter_replication.replicated_at_leaf_buffers().begin(),
      parameter_replication.replicated_at_leaf_buffers().end());
}

StatusOr<std::vector<ReplicaGroup>> HloParserImpl::ParseReplicaGroupsOnly() {
  lexer_.Lex();
  std::vector<ReplicaGroup> replica_groups;
  if (!ParseReplicaGroupsOnly(&replica_groups)) {
    return InvalidArgument("Syntax error:\n%s", GetError());
  }
  if (lexer_.GetKind() != TokKind::kEof) {
    return InvalidArgument("Syntax error:\nExtra content after replica groups");
  }
  return replica_groups;
}

StatusOr<Window> HloParserImpl::ParseWindowOnly() {
  lexer_.Lex();
  Window window;
  if (!ParseWindow(&window, /*expect_outer_curlies=*/false)) {
    return InvalidArgument("Syntax error:\n%s", GetError());
  }
  if (lexer_.GetKind() != TokKind::kEof) {
    return InvalidArgument("Syntax error:\nExtra content after window");
  }
  return window;
}

StatusOr<ConvolutionDimensionNumbers>
HloParserImpl::ParseConvolutionDimensionNumbersOnly() {
  lexer_.Lex();
  ConvolutionDimensionNumbers dnums;
  if (!ParseConvolutionDimensionNumbers(&dnums)) {
    return InvalidArgument("Syntax error:\n%s", GetError());
  }
  if (lexer_.GetKind() != TokKind::kEof) {
    return InvalidArgument(
        "Syntax error:\nExtra content after convolution dnums");
  }
  return dnums;
}

StatusOr<PaddingConfig> HloParserImpl::ParsePaddingConfigOnly() {
  lexer_.Lex();
  PaddingConfig padding_config;
  if (!ParsePaddingConfig(&padding_config)) {
    return InvalidArgument("Syntax error:\n%s", GetError());
  }
  if (lexer_.GetKind() != TokKind::kEof) {
    return InvalidArgument("Syntax error:\nExtra content after PaddingConfig");
  }
  return padding_config;
}

bool HloParserImpl::ParseSingleInstruction(HloModule* module) {
  if (create_missing_instruction_ != nullptr || !scoped_name_tables_.empty()) {
    LOG(FATAL) << "Parser state is not clean. Please do not call any other "
                  "methods before calling ParseSingleInstruction.";
  }
  HloComputation::Builder builder(module->name());

  // The missing instruction hook we register creates the shaped instruction on
  // the fly as a parameter and returns it.
  int64 parameter_count = 0;
  create_missing_instruction_ =
      [this, &builder, &parameter_count](
          const std::string& name,
          const Shape& shape) -> std::pair<HloInstruction*, LocTy>* {
    std::string new_name = name.empty() ? StrCat("_", parameter_count) : name;
    HloInstruction* parameter = builder.AddInstruction(
        HloInstruction::CreateParameter(parameter_count++, shape, new_name));
    current_name_table()[new_name] = {parameter, lexer_.GetLoc()};
    return tensorflow::gtl::FindOrNull(current_name_table(), new_name);
  };

  // Parse the instruction with the registered hook.
  Scope scope(&scoped_name_tables_);
  if (CanBeShape()) {
    // This means that the instruction's left-hand side is probably omitted,
    // e.g.
    //
    //  f32[10] fusion(...), calls={...}
    if (!ParseInstructionRhs(&builder, module->name(), lexer_.GetLoc())) {
      return false;
    }
  } else {
    // This means that the instruction's left-hand side might exist, e.g.
    //
    //  foo = f32[10] fusion(...), calls={...}
    std::string root_name;
    if (!ParseInstruction(&builder, &root_name)) {
      return false;
    }
  }

  if (lexer_.GetKind() != TokKind::kEof) {
    Error(
        lexer_.GetLoc(),
        "Syntax error:\nExpected eof after parsing single instruction.  Did "
        "you mean to write an HLO module and forget the \"HloModule\" header?");
    return false;
  }

  module->AddEntryComputation(builder.Build());
  for (auto& comp : computations_) {
    module->AddEmbeddedComputation(std::move(comp));
  }
  TF_CHECK_OK(module->set_schedule(ScheduleFromInstructionOrder(module)));
  return true;
}

}  // namespace

StatusOr<std::unique_ptr<HloModule>> ParseAndReturnUnverifiedModule(
    absl::string_view str, const HloModuleConfig& config) {
  auto module = absl::make_unique<HloModule>(/*name=*/"_", config);
  HloParserImpl parser(str);
  TF_RETURN_IF_ERROR(parser.Run(module.get()));
  return std::move(module);
}

StatusOr<std::unique_ptr<HloModule>> ParseAndReturnUnverifiedModule(
    absl::string_view str) {
  return ParseAndReturnUnverifiedModule(str, HloModuleConfig());
}

StatusOr<HloSharding> ParseSharding(absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseShardingOnly();
}

StatusOr<FrontendAttributes> ParseFrontendAttributes(absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseFrontendAttributesOnly();
}

StatusOr<std::vector<bool>> ParseParameterReplication(absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseParameterReplicationOnly();
}

StatusOr<std::vector<ReplicaGroup>> ParseReplicaGroupsOnly(
    absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseReplicaGroupsOnly();
}

StatusOr<Window> ParseWindow(absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseWindowOnly();
}

StatusOr<ConvolutionDimensionNumbers> ParseConvolutionDimensionNumbers(
    absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseConvolutionDimensionNumbersOnly();
}

StatusOr<PaddingConfig> ParsePaddingConfig(absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParsePaddingConfigOnly();
}

StatusOr<Shape> ParseShape(absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseShapeOnly();
}

std::unique_ptr<HloParser> HloParser::CreateHloParserForTests(
    absl::string_view str) {
  return absl::make_unique<HloParserImpl>(str);
}

}  // namespace xla
