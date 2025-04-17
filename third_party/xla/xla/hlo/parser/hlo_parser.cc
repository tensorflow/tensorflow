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

#include "xla/hlo/parser/hlo_parser.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "xla/array.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_domain_metadata.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/hlo_sharding_metadata.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/hlo/parser/hlo_lexer.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/name_uniquer.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/gtl/map_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

namespace xla {

namespace {

using absl::StrAppend;
using absl::StrCat;
using absl::StrFormat;
using absl::StrJoin;
using std::nullopt;
using std::optional;

// VLOG levels for debug and error messages.
const int8_t kDebugLevel = 10;
const int8_t kErrorLevel = 1;

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

bool CanInferShape(HloOpcode code) {
  switch (code) {
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kAddDependency:
    case HloOpcode::kAfterAll:
    case HloOpcode::kAtan2:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCall:
    case HloOpcode::kCeil:
    case HloOpcode::kCholesky:
    case HloOpcode::kClamp:
    case HloOpcode::kClz:
    case HloOpcode::kCompare:
    case HloOpcode::kComplex:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConditional:
    case HloOpcode::kConvolution:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kDivide:
    case HloOpcode::kDomain:
    case HloOpcode::kDot:
    case HloOpcode::kErf:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFft:
    case HloOpcode::kFloor:
    case HloOpcode::kGather:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kLogistic:
    case HloOpcode::kAnd:
    case HloOpcode::kNot:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kMap:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kPad:
    case HloOpcode::kPartitionId:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kPower:
    case HloOpcode::kRaggedDot:
    case HloOpcode::kReal:
    case HloOpcode::kReduce:
    case HloOpcode::kRemainder:
    case HloOpcode::kReplicaId:
    case HloOpcode::kReverse:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kRsqrt:
    case HloOpcode::kScatter:
    case HloOpcode::kSelect:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSort:
    case HloOpcode::kSubtract:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    case HloOpcode::kTranspose:
    case HloOpcode::kTriangularSolve:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
    case HloOpcode::kTopK:
      return true;
    // Technically the following ops do not require an explicit result shape,
    // but we made it so that we always write the shapes explicitly.
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectiveBroadcast:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCopyDone:
    case HloOpcode::kCopyStart:
    case HloOpcode::kDynamicReshape:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kRaggedAllToAll:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kSlice:
    // The following ops require an explicit result shape.
    case HloOpcode::kBitcast:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kConstant:
    case HloOpcode::kConvert:
    case HloOpcode::kCustomCall:
    case HloOpcode::kFusion:
    case HloOpcode::kInfeed:
    case HloOpcode::kIota:
    case HloOpcode::kOutfeed:
    case HloOpcode::kParameter:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kReshape:
    case HloOpcode::kRng:
    case HloOpcode::kRngBitGenerator:
    case HloOpcode::kRngGetAndUpdateState:
    case HloOpcode::kStochasticConvert:
      return false;
  }
}

// Parser for the HloModule::ToString() format text.
class HloParserImpl : public HloParser {
 public:
  using LocTy = HloLexer::LocTy;
  using BoolList = absl::InlinedVector<bool, 1>;

  explicit HloParserImpl(absl::string_view str,
                         const HloParserOptions& options = HloParserOptions())
      : lexer_(str), options_(options) {}

  // Runs the parser and constructs the resulting HLO in the given (empty)
  // HloModule. Returns the error status in case an error occurred.
  absl::Status Run(HloModule* module) override;

  // Returns the error information.
  std::string GetError() const { return StrJoin(error_, "\n"); }

  // Stand alone parsing utils for various aggregate data types.
  absl::StatusOr<Shape> ParseShapeOnly();
  absl::StatusOr<Layout> ParseLayoutOnly();
  absl::StatusOr<HloSharding> ParseShardingOnly();
  absl::StatusOr<FrontendAttributes> ParseFrontendAttributesOnly();
  absl::StatusOr<StatisticsViz> ParseStatisticsVizOnly();
  absl::StatusOr<std::vector<bool>> ParseParameterReplicationOnly();
  absl::StatusOr<BoolList> ParseBooleanListOrSingleBooleanOnly();
  absl::StatusOr<Window> ParseWindowOnly();
  absl::StatusOr<ConvolutionDimensionNumbers>
  ParseConvolutionDimensionNumbersOnly();
  absl::StatusOr<PaddingConfig> ParsePaddingConfigOnly();
  absl::StatusOr<std::vector<ReplicaGroup>> ParseReplicaGroupsOnly();
  absl::StatusOr<CollectiveDeviceList> ParseCollectiveDeviceListOnly();

 private:
  // Types of attributes.
  enum class AttrTy {
    kBool,
    kInt64,
    kInt32,
    kFloat,
    kString,
    kLiteral,
    kBracedInt64List,
    kBracedInt64ListList,
    kHloComputation,
    kBracedHloComputationList,
    kFftType,
    kPaddingType,
    kComparisonDirection,
    kComparisonType,
    kWindow,
    kConvolutionDimensionNumbers,
    kSharding,
    kFrontendAttributes,
    kStatisticsViz,
    kBracedBoolListOrBool,
    kParameterReplication,
    kInstructionList,
    kSliceRanges,
    kPaddingConfig,
    kMetadata,
    kFusionKind,
    kDistribution,
    kDomain,
    kPrecisionList,
    kShape,
    kShapeList,
    kEnum,
    kRandomAlgorithm,
    kPrecisionAlgorithm,
    kResultAccuracyType,
    kAliasing,
    kBufferDonor,
    kComputationLayout,
    kInstructionAliasing,
    kCustomCallSchedule,
    kCustomCallApiVersion,
    kSparsityDescriptor,
    // A double-quoted string, or a string that looks like a JSON dictionary
    // enclosed in matching curly braces (returned value includes the curlies).
    kStringOrJsonDict,
    kCollectiveDeviceList,
    kResultAccuracy,
    kOriginalValue,
  };

  struct AttrConfig {
    bool required;     // whether it's required or optional
    AttrTy attr_type;  // what type it is
    void* result;      // where to store the parsed result.
  };

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
  // if `parse_module_without_header` is true, the parsed text is sequence of
  // computations, and assume computation with `ENTRY` annotation or the last
  // computation as module's entry computation, also using the entry
  // computation's parameter and `ROOT` instruction's layout as module's layout.
  bool ParseHloModule(HloModule* module,
                      bool parse_module_without_header = false);

  bool ParseComputations(HloModule* module);
  bool ParseComputation(HloComputation** entry_computation);
  bool ParseInstructionList(HloComputation** computation,
                            const std::string& computation_name);
  bool ParseInstruction(HloComputation::Builder* builder,
                        std::string* root_name);
  bool ParseInstructionRhs(HloComputation::Builder* builder, std::string name,
                           LocTy name_loc, bool allow_attributes = true);
  bool ParseControlPredecessors(HloInstruction* instruction);
  bool ParseLiteral(Literal* literal);
  bool ParseLiteral(Literal* literal, const Shape& shape);
  bool ParseTupleLiteral(Literal* literal, const Shape& shape);
  bool ParseNonTupleLiteral(Literal* literal, const Shape& shape);
  bool ParseDenseLiteral(Literal* literal, const Shape& shape);

  // Parses and creates instruction given name, shape, opcode etc. This is
  // refactored out from ParseInstructionRhs to allow recursion of wrapped
  // async instructions to allow parsing for wrapped-op-specific attributes.
  HloInstruction* CreateInstruction(
      HloComputation::Builder* builder, absl::string_view name,
      std::optional<Shape> shape, HloOpcode opcode,
      std::optional<HloOpcode> async_wrapped_opcode,
      absl::flat_hash_map<std::string, AttrConfig>& attrs,
      bool allow_attributes,
      std::vector<HloInstruction*>* preset_operands = nullptr);

  // Sets the sub-value of literal at the given linear index to the
  // given value. If the literal is dense, it must have the default layout.
  //
  // `loc` should be the source location of the value.
  bool SetValueInLiteral(LocTy loc, int64_t value, int64_t index,
                         Literal* literal);
  bool SetValueInLiteral(LocTy loc, double value, int64_t index,
                         Literal* literal);
  bool SetValueInLiteral(LocTy loc, bool value, int64_t index,
                         Literal* literal);
  bool SetValueInLiteral(LocTy loc, std::complex<double> value, int64_t index,
                         Literal* literal);
  // `loc` should be the source location of the value.
  template <typename LiteralNativeT, typename ParsedElemT>
  bool SetValueInLiteralHelper(LocTy loc, ParsedElemT value, int64_t index,
                               Literal* literal);

  // Checks whether the given value is within the range of LiteralNativeT.
  // `loc` should be the source location of the value.
  template <typename LiteralNativeT, typename ParsedElemT>
  bool CheckParsedValueIsInRange(LocTy loc, ParsedElemT value);
  template <typename LiteralNativeT>
  bool CheckParsedValueIsInRange(LocTy loc, std::complex<double> value);

  bool ParseOperands(std::vector<HloInstruction*>* operands,
                     HloComputation::Builder* builder);
  // Fills parsed operands into 'operands' and expects a certain number of
  // operands.
  bool ParseOperands(std::vector<HloInstruction*>* operands,
                     HloComputation::Builder* builder, int expected_size);

  // Describes the start, limit, and stride on every dimension of the operand
  // being sliced.
  struct SliceRanges {
    std::vector<int64_t> starts;
    std::vector<int64_t> limits;
    std::vector<int64_t> strides;
  };

  // The data parsed for the kDomain instruction.
  struct DomainData {
    std::unique_ptr<DomainMetadata> entry_metadata;
    std::unique_ptr<DomainMetadata> exit_metadata;
  };

  // attributes ::= (',' attribute)*
  //
  // Parses attributes given names and configs of the attributes. Each parsed
  // result is passed back through the result pointer in corresponding
  // AttrConfig. Note that the result pointer must point to a optional<T> typed
  // variable which outlives this function. Returns false on error. You should
  // not use the any of the results if this function failed.
  //
  // If allow_attributes is false, returns an error if any attributes are
  // present.  This is used for contexts in which attributes are not allowed but
  // e.g. we *also* want to raise an error if any required attributes are
  // missing.
  //
  // Example usage:
  //
  //  absl::flat_hash_map<std::string, AttrConfig> attrs;
  //  optional<int64_t> foo;
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
      const absl::flat_hash_map<std::string, AttrConfig>& attrs,
      bool allow_attributes = true, const std::optional<Shape>& shape = {});

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
      absl::flat_hash_set<std::string>* seen_attrs,
      const std::optional<Shape>& shape = {});

  // Copy attributes from `attrs` to `message`, unless the attribute name is in
  // `non_proto_attrs`.
  bool CopyAttributeToProtoMessage(
      absl::flat_hash_set<std::string> non_proto_attrs,
      const absl::flat_hash_map<std::string, AttrConfig>& attrs,
      tsl::protobuf::Message* message);

  // Parses an attribute string into a protocol buffer `message`.
  // Since proto3 has no notion of mandatory fields, `required_attrs` gives the
  // set of mandatory attributes.
  // `non_proto_attrs` specifies attributes that are not written to the proto,
  // but added to the HloInstruction.
  bool ParseAttributesAsProtoMessage(
      const absl::flat_hash_map<std::string, AttrConfig>& non_proto_attrs,
      tsl::protobuf::Message* message);

  // Parses a name and finds the corresponding hlo computation.
  bool ParseComputationName(HloComputation** value);
  // Parses a list of names and finds the corresponding hlo instructions.
  bool ParseInstructionNames(std::vector<HloInstruction*>* instructions);
  // Pass expect_outer_curlies == true when parsing a Window in the context of a
  // larger computation.  Pass false when parsing a stand-alone Window string.
  bool ParseWindow(Window* window, bool expect_outer_curlies);
  bool ParseConvolutionDimensionNumbers(ConvolutionDimensionNumbers* dnums);
  bool ParsePaddingConfig(PaddingConfig* padding);
  bool ParseMetadata(OpMetadata& metadata);
  bool ParseSingleOrListMetadata(std::vector<OpMetadata>& metadata);
  bool ParseOpShardingType(OpSharding::Type* type);
  bool ParseListShardingType(std::vector<OpSharding::Type>* types);
  bool ParseSharding(std::optional<HloSharding>& sharding);
  bool ParseCollectiveDeviceList(CollectiveDeviceList* device_list);
  bool ParseFrontendAttributes(FrontendAttributes* frontend_attributes);
  bool ParseStatisticsViz(StatisticsViz* statistics_viz);
  bool ParseTileAssignment(std::vector<int64_t>& tile_assignment_dimensions,
                           std::vector<int64_t>& iota_reshape_dims,
                           std::vector<int>& iota_transpose_perm,
                           std::vector<int64_t>* devices);
  bool ParseSingleSharding(std::optional<HloSharding>& sharding,
                           bool lbrace_pre_lexed);
  bool ParseParameterReplication(ParameterReplication* parameter_replication);
  bool ParseBooleanListOrSingleBoolean(BoolList* boolean_list);
  bool ParseReplicaGroupsOnly(std::vector<ReplicaGroup>* replica_groups);

  // Parses the metadata behind a kDOmain instruction.
  bool ParseDomain(DomainData* domain);

  // Parses a sub-attribute of the window attribute, e.g.,size=1x2x3.
  bool ParseDxD(const std::string& name, std::vector<int64_t>* result);
  // Parses window's pad sub-attribute, e.g., pad=0_0x3x3.
  bool ParseWindowPad(std::vector<std::vector<int64_t>>* pad);

  bool ParseSliceRanges(SliceRanges* result);
  bool ParsePrecisionList(std::vector<PrecisionConfig::Precision>* result);
  bool ParseHloComputation(HloComputation** result);
  bool ParseHloComputationList(std::vector<HloComputation*>* result);
  bool ParseShapeList(std::vector<Shape>* result);
  bool ParseInt64List(TokKind start, TokKind end, TokKind delim,
                      std::vector<int64_t>* result);
  bool ParseInt64ListList(TokKind start, TokKind end, TokKind delim,
                          std::vector<std::vector<int64_t>>* result);
  // 'parse_and_add_item' is an lambda to parse an element in the list and add
  // the parsed element to the result. It's supposed to capture the result.
  bool ParseList(TokKind start, TokKind end, TokKind delim,
                 absl::FunctionRef<bool()> parse_and_add_item);

  bool ParseParamListToShape(Shape* shape, LocTy* shape_loc);
  bool ParseParamList();
  bool ParseName(std::string* result);
  bool ParseAttributeName(std::string* result);
  bool ParseString(std::string* result);
  bool ParseJsonDict(std::string* result);
  bool ParseDimensionSizes(std::vector<int64_t>* dimension_sizes,
                           std::vector<bool>* dynamic_dimensions);
  bool ParseShape(Shape* result, bool allow_fallback_to_default_layout = true);
  bool ParseLayout(Layout* layout);
  bool ParseLayoutIntAttribute(int64_t* attr_value,
                               absl::string_view attr_description);
  bool ParseDimLevelTypes(
      absl::InlinedVector<DimLevelType, InlineRank()>* dim_level_types,
      absl::InlinedVector<bool, InlineRank()>* dim_unique,
      absl::InlinedVector<bool, InlineRank()>* dim_ordered);
  bool ParseTiles(std::vector<Tile>* tiles);
  bool ParseSplitConfigs(std::vector<SplitConfig>& split_configs);
  bool ParsePhysicalShape(Shape* physical_shape);
  bool ParseOpcode(HloOpcode* opcode,
                   std::optional<HloOpcode>* async_wrapped_opcode);
  bool ParseFftType(FftType* result);
  bool ParsePaddingType(PaddingType* result);
  bool ParsePrimitiveType(PrimitiveType* result);
  bool ParseComparisonDirection(ComparisonDirection* result);
  bool ParseComparisonType(Comparison::Type* result);
  bool ParseFusionKind(HloInstruction::FusionKind* result);
  bool ParseRandomDistribution(RandomDistribution* result);
  bool ParseRandomAlgorithm(RandomAlgorithm* result);
  bool ParsePrecision(PrecisionConfig::Precision* result);
  bool ParseAlgorithm(PrecisionConfig::Algorithm* result);
  bool ParseResultAccuracyType(ResultAccuracy::Mode* result);
  bool ParseResultAccuracyTolerance(ResultAccuracy::Tolerance* result);
  bool ParseResultAccuracy(ResultAccuracy* result);
  bool ParseInt64(int64_t* result);
  bool ParseDouble(double* result);
  bool ParseComplex(std::complex<double>* result);
  bool ParseBool(bool* result);
  bool ParseToken(TokKind kind, const std::string& msg);
  bool ParseUnsignedIntegerType(PrimitiveType* primitive_type);
  bool ParseOriginalValue(
      optional<std::shared_ptr<OriginalValue>>* original_value,
      const Shape& shape);

  using AliasingData =
      absl::flat_hash_map<ShapeIndex, HloInputOutputAliasConfig::Alias>;
  using BufferDonor = absl::flat_hash_set<HloBufferDonorConfig::BufferDonor>;

  // Parses the aliasing and buffer_donor information from string `s`, returns
  // `false` if it fails.
  bool ParseAliasing(AliasingData* data);
  bool ParseBufferDonor(BufferDonor* data);

  // Parses the entry computation layout.
  bool ParseComputationLayout(ComputationLayout* computation_layout);

  // Parses the per-instruction aliasing information from string `s`, returns
  // `false` if it fails.
  bool ParseInstructionOutputOperandAliasing(
      std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>*
          aliasing_output_operand_pairs);

  bool ParseCustomCallSchedule(CustomCallSchedule* result);
  bool ParseCustomCallApiVersion(CustomCallApiVersion* result);
  bool ParseSparsityDescriptor(std::vector<SparsityDescriptor>* result);
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

  // Used to generate names for anonymous instructions.
  NameUniquer name_uniquer_{/*separator=*/"."};

  const HloParserOptions options_;
};

bool SplitToInt64s(absl::string_view s, char delim, std::vector<int64_t>* out) {
  for (const auto& split : absl::StrSplit(s, delim)) {
    int64_t val;
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
    absl::Span<const std::vector<int64_t>> groups) {
  std::vector<ReplicaGroup> replica_groups;
  absl::c_transform(groups, std::back_inserter(replica_groups),
                    [](const std::vector<int64_t>& ids) {
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
  VLOG(kErrorLevel) << "Error: " << error_.back();
  return false;
}

bool HloParserImpl::TokenError(absl::string_view msg) {
  return Error(lexer_.GetLoc(), msg);
}

absl::Status HloParserImpl::Run(HloModule* module) {
  lexer_.Lex();
  if ((lexer_.GetKind() == TokKind::kw_HloModule) ||
      (lexer_.GetKind() == TokKind::kw_ENTRY) ||
      (lexer_.LookAhead() == TokKind::kLbrace)) {
    // This means that the text contains a full HLO module.
    bool parse_module_without_header =
        (lexer_.GetKind() == TokKind::kw_HloModule) ? false : true;
    if (!ParseHloModule(module, parse_module_without_header)) {
      return InvalidArgument(
          "Syntax error when trying to parse the text as a HloModule:\n%s",
          GetError());
    }
    return absl::OkStatus();
  }
  // This means that the text is a single HLO instruction.
  if (!ParseSingleInstruction(module)) {
    return InvalidArgument(
        "Syntax error when trying to parse the text as a single "
        "HloInstruction:\n%s",
        GetError());
  }
  return absl::OkStatus();
}

std::pair<HloInstruction*, HloParserImpl::LocTy>*
HloParserImpl::FindInstruction(const std::string& name,
                               const optional<Shape>& shape) {
  std::pair<HloInstruction*, LocTy>* instr = nullptr;
  if (!name.empty()) {
    instr = tsl::gtl::FindOrNull(current_name_table(), name);
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

  std::vector<int64_t> idxs;
  while (lexer_.GetKind() != TokKind::kRbrace) {
    int64_t idx;
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
        "<input_param_shape_index>) OR <output_shape_index>: <input_param>";
    if (!ParseToken(TokKind::kColon, errmsg)) {
      return false;
    }

    if (!ParseToken(TokKind::kLparen, errmsg)) {
      return false;
    }
    int64_t param_num;
    ParseInt64(&param_num);
    if (!ParseToken(TokKind::kComma, errmsg)) {
      return false;
    }
    ShapeIndex param_idx;
    if (!ParseShapeIndex(&param_idx)) {
      return false;
    }

    HloInputOutputAliasConfig::AliasKind alias_kind =
        HloInputOutputAliasConfig::kMayAlias;
    if (EatIfPresent(TokKind::kComma)) {
      std::string type;
      ParseName(&type);
      if (type == "must-alias") {
        alias_kind = HloInputOutputAliasConfig::kMustAlias;
      } else if (type == "may-alias") {
        alias_kind = HloInputOutputAliasConfig::kMayAlias;
      } else {
        return TokenError("Unexpected aliasing kind; expected SYSTEM or USER");
      }
    }

    data->emplace(std::piecewise_construct, std::forward_as_tuple(out),
                  std::forward_as_tuple(param_num, param_idx, alias_kind));
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

bool HloParserImpl::ParseBufferDonor(BufferDonor* data) {
  if (!ParseToken(TokKind::kLbrace,
                  "Expects '{' at the start of buffer donor description")) {
    return false;
  }

  std::string errmsg =
      "Expected format: (<input_param>, <input_param_shape_index>)";
  while (lexer_.GetKind() != TokKind::kRbrace) {
    if (!ParseToken(TokKind::kLparen, errmsg)) {
      return false;
    }

    int64_t param_num;
    ParseInt64(&param_num);

    if (!ParseToken(TokKind::kComma, errmsg)) {
      return false;
    }

    ShapeIndex param_idx;
    if (!ParseShapeIndex(&param_idx)) {
      return false;
    }

    if (!ParseToken(TokKind::kRparen, errmsg)) {
      return false;
    }

    data->emplace(param_num, param_idx);

    if (!EatIfPresent(TokKind::kComma)) {
      break;
    }
  }
  if (!ParseToken(TokKind::kRbrace,
                  "Expects '}' at the end of buffer donor description")) {
    return false;
  }
  return true;
}

bool HloParserImpl::ParseComputationLayout(
    ComputationLayout* computation_layout) {
  if (!ParseToken(TokKind::kLbrace,
                  "Expects '{' at the start of aliasing description")) {
    return false;
  }
  if (!ParseToken(TokKind::kLparen, "Expects ( before parameter shape list")) {
    return false;
  }
  while (lexer_.GetKind() != TokKind::kRparen) {
    Shape param;
    if (!ParseShape(&param,
                    /* allow_fallback_to_default_layout=*/
                    !options_.keep_module_auto_layouts())) {
      return false;
    }
    computation_layout->add_parameter_layout(ShapeLayout(param));
    if (lexer_.GetKind() == TokKind::kRparen) {
      break;
    }
    if (!ParseToken(TokKind::kComma, "Expects , between parameter shapes")) {
      return false;
    }
  }

  if (!ParseToken(TokKind::kRparen,
                  "Expects ) at end of parameter shape list")) {
    return false;
  }
  if (!ParseToken(TokKind::kArrow, "Expects -> before result shape")) {
    return false;
  }
  Shape result;
  if (!ParseShape(&result,
                  /* allow_fallback_to_default_layout=*/
                  !options_.keep_module_auto_layouts())) {
    return false;
  }
  *computation_layout->mutable_result_layout() = ShapeLayout(result);
  if (!ParseToken(TokKind::kRbrace,
                  "Expects '}' at the end of computation layouts")) {
    return false;
  }
  return true;
}

bool HloParserImpl::ParseInstructionOutputOperandAliasing(
    std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>*
        aliasing_output_operand_pairs) {
  if (!ParseToken(
          TokKind::kLbrace,
          "Expects '{' at the start of instruction aliasing description")) {
    return false;
  }

  while (lexer_.GetKind() != TokKind::kRbrace) {
    ShapeIndex out;
    if (!ParseShapeIndex(&out)) {
      return false;
    }
    std::string errmsg =
        "Expected format: <output_shape_index>: (<operand_index>, "
        "<operand_shape_index>)";
    if (!ParseToken(TokKind::kColon, errmsg)) {
      return false;
    }

    if (!ParseToken(TokKind::kLparen, errmsg)) {
      return false;
    }
    int64_t operand_index;
    ParseInt64(&operand_index);
    if (!ParseToken(TokKind::kComma, errmsg)) {
      return false;
    }
    ShapeIndex operand_shape_index;
    if (!ParseShapeIndex(&operand_shape_index)) {
      return false;
    }

    aliasing_output_operand_pairs->emplace_back(
        out,
        std::pair<int64_t, ShapeIndex>{operand_index, operand_shape_index});
    if (!ParseToken(TokKind::kRparen, errmsg)) {
      return false;
    }

    if (!EatIfPresent(TokKind::kComma)) {
      break;
    }
  }
  if (!ParseToken(
          TokKind::kRbrace,
          "Expects '}' at the end of instruction aliasing description")) {
    return false;
  }
  return true;
}

bool HloParserImpl::ParseCustomCallSchedule(CustomCallSchedule* result) {
  VLOG(kDebugLevel) << "ParseCustomCallSchedule";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects custom-call schedule");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToCustomCallSchedule(val);
  if (!status_or_result.ok()) {
    return TokenError(
        StrFormat("expects custom-call schedule but sees: %s, error: %s", val,
                  status_or_result.status().message()));
  }
  *result = status_or_result.value();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseCustomCallApiVersion(CustomCallApiVersion* result) {
  VLOG(kDebugLevel) << "ParseCustomCallApiVersion";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects custom-call API version");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToCustomCallApiVersion(val);
  if (!status_or_result.ok()) {
    return TokenError(
        StrFormat("expects custom-call API version but sees: %s, error: %s",
                  val, status_or_result.status().message()));
  }
  *result = status_or_result.value();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseSparsityDescriptor(
    std::vector<SparsityDescriptor>* result) {
  VLOG(kDebugLevel) << "ParseSparsityDescriptor";
  if (lexer_.GetKind() != TokKind::kSparsityDesc) {
    return TokenError("expects sparsity descriptor, e.g. L.0@2:4");
  }
  std::string val = lexer_.GetStrVal();
  std::vector<absl::string_view> split = absl::StrSplit(val, '_');
  for (absl::string_view item : split) {
    std::vector<absl::string_view> splitA = absl::StrSplit(item, '@');
    std::vector<absl::string_view> splitB = absl::StrSplit(splitA[0], '.');
    std::vector<absl::string_view> splitC = absl::StrSplit(splitA[1], ':');
    SparsityDescriptor descriptor;
    int dim, n, m;
    if (!absl::SimpleAtoi(splitB[1], &dim) || dim < 0) {
      return TokenError("Invalid dimension number");
    }
    if (!absl::SimpleAtoi(splitC[0], &n) || !absl::SimpleAtoi(splitC[1], &m) ||
        n < 1 || m <= n) {
      return TokenError("Invalid structured sparsity type");
    }
    descriptor.set_type(SparsityType::SPARSITY_STRUCTURED_N_M);
    descriptor.set_index(splitB[0] == "L" ? 0 : 1);
    descriptor.set_dimension(dim);
    descriptor.set_n(n);
    descriptor.set_m(m);
    result->push_back(descriptor);
  }
  lexer_.Lex();
  return true;
}

// ::= 'HloModule' name computations
bool HloParserImpl::ParseHloModule(HloModule* module,
                                   bool parse_module_without_header) {
  std::string name;
  std::optional<bool> is_scheduled;
  std::optional<int64_t> replica_count;
  std::optional<int64_t> num_partitions;
  std::optional<AliasingData> aliasing_data;
  std::optional<BufferDonor> buffer_donor_data;
  std::optional<bool> alias_passthrough_params;
  absl::flat_hash_map<std::string, AttrConfig> attrs;
  std::optional<ComputationLayout> entry_computation_layout;
  std::optional<FrontendAttributes> frontend_attributes;
  BoolList allow_spmd_sharding_propagation_to_parameters;
  BoolList allow_spmd_sharding_propagation_to_output;

  attrs["is_scheduled"] = {/*required=*/false, AttrTy::kBool, &is_scheduled};
  attrs["replica_count"] = {/*required=*/false, AttrTy::kInt64, &replica_count};
  attrs["num_partitions"] = {/*required=*/false, AttrTy::kInt64,
                             &num_partitions};
  attrs["input_output_alias"] = {/*required=*/false, AttrTy::kAliasing,
                                 &aliasing_data};
  attrs["buffer_donor"] = {/*required=*/false, AttrTy::kBufferDonor,
                           &buffer_donor_data};
  attrs["alias_passthrough_params"] = {/*required=*/false, AttrTy::kBool,
                                       &alias_passthrough_params};
  attrs["entry_computation_layout"] = {/*required=*/false,
                                       AttrTy::kComputationLayout,
                                       &entry_computation_layout};
  attrs["frontend_attributes"] = {
      /*required=*/false, AttrTy::kFrontendAttributes, &frontend_attributes};
  attrs["allow_spmd_sharding_propagation_to_parameters"] = {
      /*required=*/false, AttrTy::kBracedBoolListOrBool,
      &allow_spmd_sharding_propagation_to_parameters};
  attrs["allow_spmd_sharding_propagation_to_output"] = {
      /*required=*/false, AttrTy::kBracedBoolListOrBool,
      &allow_spmd_sharding_propagation_to_output};

  if (!parse_module_without_header) {
    if (lexer_.GetKind() != TokKind::kw_HloModule) {
      return TokenError("expects HloModule");
    }
    // Eat 'HloModule'
    lexer_.Lex();

    if (!ParseName(&name)) {
      return false;
    }
    if (!ParseAttributes(attrs)) {
      return false;
    }
    module->set_name(name);
  }

  if (!ParseComputations(module)) {
    return false;
  }

  if (parse_module_without_header) {
    name = absl::StrCat("module_", module->entry_computation()->name());
  }

  module->set_name(name);

  if (is_scheduled.value_or(false)) {
    TF_CHECK_OK(module->set_schedule(ScheduleFromInstructionOrder(module)));
  }
  HloModuleConfig config = module->config();
  bool default_config = true;
  if (alias_passthrough_params.value_or(false)) {
    config.set_alias_passthrough_params(true);
    default_config = false;
  }
  if (num_partitions.value_or(1) != 1) {
    config.set_num_partitions(*num_partitions);
    config.set_use_spmd_partitioning(true);
    default_config = false;
  }
  if (replica_count.value_or(1) != 1) {
    config.set_replica_count(*replica_count);
    default_config = false;
  }
  if (entry_computation_layout.has_value()) {
    *config.mutable_entry_computation_layout() = *entry_computation_layout;
    default_config = false;
  } else {
    // If entry_computation_layout is not specified explicitly, we infer the
    // layout from parameter and root instructions.
    HloComputation* entry_computation = module->entry_computation();
    for (int64_t p = 0; p < entry_computation->num_parameters(); p++) {
      const Shape& param_shape =
          entry_computation->parameter_instruction(p)->shape();
      TF_CHECK_OK(module->mutable_entry_computation_layout()
                      ->mutable_parameter_layout(p)
                      ->CopyLayoutFromShape(param_shape));
    }
    const Shape& result_shape = entry_computation->root_instruction()->shape();
    TF_CHECK_OK(module->mutable_entry_computation_layout()
                    ->mutable_result_layout()
                    ->CopyLayoutFromShape(result_shape));
  }
  if (frontend_attributes) {
    module->set_frontend_attributes(frontend_attributes.value());
  }
  if (!allow_spmd_sharding_propagation_to_parameters.empty()) {
    config.set_allow_spmd_sharding_propagation_to_parameters(
        allow_spmd_sharding_propagation_to_parameters);
    default_config = false;
  }
  if (!allow_spmd_sharding_propagation_to_output.empty()) {
    config.set_allow_spmd_sharding_propagation_to_output(
        allow_spmd_sharding_propagation_to_output);
    default_config = false;
  }
  if (!default_config) {
    module->set_config(config);
  }
  if (aliasing_data) {
    HloInputOutputAliasConfig alias_config(module->result_shape());
    for (auto& p : *aliasing_data) {
      absl::Status st =
          alias_config.SetUpAlias(p.first, p.second.parameter_number,
                                  p.second.parameter_index, p.second.kind);
      if (!st.ok()) {
        return TokenError(st.message());
      }
    }
    module->input_output_alias_config() = alias_config;
  }
  if (buffer_donor_data) {
    HloBufferDonorConfig buffer_donor_config;
    for (auto& p : *buffer_donor_data) {
      absl::Status st =
          buffer_donor_config.AddBufferDonor(p.param_number, p.param_index);
      if (!st.ok()) {
        return TokenError(st.message());
      }
    }
    module->buffer_donor_config() = buffer_donor_config;
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
    module->AddEntryComputation(std::move(computations_[i]));
  }
  return true;
}

// computation ::= ('ENTRY')? name (param_list_to_shape)? instruction_list(,
// 'execution_thread='execution_thread)?
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
  absl::flat_hash_map<std::string, AttrConfig> attrs;
  optional<std::string> execution_thread = HloInstruction::kMainExecutionThread;
  attrs["execution_thread"] = {/*required=*/false, AttrTy::kString,
                               &execution_thread};
  if (!ParseAttributes(attrs)) {
    return false;
  }
  computation->SetExecutionThread(*execution_thread);
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
        tsl::gtl::FindOrNull(current_name_table(), root_name);

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
                                        std::string name, LocTy name_loc,
                                        bool allow_attributes) {
  Shape shape;
  HloOpcode opcode;
  std::optional<HloOpcode> async_wrapped_opcode;
  std::vector<HloInstruction*> operands;

  const bool parse_shape = CanBeShape();
  if ((parse_shape && !ParseShape(&shape)) ||
      !ParseOpcode(&opcode, &async_wrapped_opcode)) {
    return false;
  }
  if (!parse_shape && !CanInferShape(opcode)) {
    return TokenError(StrFormat("cannot infer shape for opcode: %s",
                                HloOpcodeString(opcode)));
  }

  // Add optional attributes. These are added to any HloInstruction type if
  // present.
  absl::flat_hash_map<std::string, AttrConfig> attrs;
  optional<HloSharding> sharding;
  optional<FrontendAttributes> frontend_attributes;
  optional<StatisticsViz> statistics_viz;
  attrs["sharding"] = {/*required=*/false, AttrTy::kSharding, &sharding};
  attrs["frontend_attributes"] = {
      /*required=*/false, AttrTy::kFrontendAttributes, &frontend_attributes};
  attrs["statistics"] = {/*required=*/false, AttrTy::kStatisticsViz,
                         &statistics_viz};
  optional<ParameterReplication> parameter_replication;
  attrs["parameter_replication"] = {/*required=*/false,
                                    AttrTy::kParameterReplication,
                                    &parameter_replication};
  optional<std::vector<HloInstruction*>> predecessors;
  attrs["control-predecessors"] = {/*required=*/false, AttrTy::kInstructionList,
                                   &predecessors};

  optional<std::shared_ptr<OriginalValue>> original_value;
  attrs["origin"] = {/*required=*/false, AttrTy::kOriginalValue,
                     &original_value};

  optional<OpMetadata> metadata;
  attrs["metadata"] = {/*required=*/false, AttrTy::kMetadata, &metadata};

  optional<std::string> backend_config;
  attrs["backend_config"] = {/*required=*/false, AttrTy::kStringOrJsonDict,
                             &backend_config};

  std::optional<Shape> maybe_shape;
  if (parse_shape) {
    maybe_shape = shape;
  }
  HloInstruction* instruction =
      CreateInstruction(builder, name, maybe_shape, opcode,
                        async_wrapped_opcode, attrs, allow_attributes);
  if (instruction == nullptr) {
    return false;
  }

  // Generate a unique name if the name is empty.  This is used for nested
  // instructions (e.g. the `max` in add(max(x, y), z)).
  //
  // Otherwise, register the given name with the name uniquer.
  if (name.empty()) {
    name = name_uniquer_.GetUniqueName(
        absl::StrCat(HloOpcodeString(instruction->opcode()), ".anon"));
  } else {
    name_uniquer_.GetUniqueName(name);
  }

  instruction->SetAndSanitizeName(name);
  if (instruction->name() != name) {
    return Error(name_loc,
                 StrCat("illegal instruction name: ", name,
                        "; suggest renaming to: ", instruction->name()));
  }

  // Add shared attributes like metadata to the instruction, if they were seen.
  if (sharding) {
    // TODO(b/257495070): Eliminate tuple sharding normalization in HLO parser.
    // Allow existing HLO text with invalid sharding on tuple shapes by
    // normalizing tuple sharding.
    instruction->set_sharding(
        sharding->NormalizeTupleSharding(instruction->shape()));
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
      absl::Status status = pre->AddControlDependencyTo(instruction);
      if (!status.ok()) {
        return Error(name_loc, StrCat("error adding control dependency for: ",
                                      name, " status: ", status.ToString()));
      }
    }
  }
  if (metadata) {
    instruction->set_metadata(*metadata);
    if (instruction->IsAsynchronous()) {
      instruction->async_wrapped_instruction()->set_metadata(*metadata);
    }
  }
  if (original_value) {
    instruction->set_original_value(*original_value);
    if (instruction->IsAsynchronous()) {
      instruction->async_wrapped_instruction()->set_original_value(
          *original_value);
    }
  }
  if (backend_config) {
    instruction->set_raw_backend_config_string(*backend_config);
    if (instruction->IsAsynchronous()) {
      instruction->async_wrapped_instruction()->set_raw_backend_config_string(
          *backend_config);
    }
  }
  if (frontend_attributes) {
    instruction->set_frontend_attributes(*frontend_attributes);
    if (instruction->IsAsynchronous()) {
      instruction->async_wrapped_instruction()->set_frontend_attributes(
          *frontend_attributes);
    }
  }
  if (statistics_viz) {
    instruction->set_statistics_viz(*statistics_viz);
    if (instruction->IsAsynchronous()) {
      instruction->async_wrapped_instruction()->set_statistics_viz(
          *statistics_viz);
    }
  }

  return AddInstruction(name, instruction, name_loc);
}

HloInstruction* HloParserImpl::CreateInstruction(  // NOLINT
    HloComputation::Builder* builder, absl::string_view name,
    std::optional<Shape> shape, HloOpcode opcode,
    std::optional<HloOpcode> async_wrapped_opcode,
    absl::flat_hash_map<std::string, AttrConfig>& attrs, bool allow_attributes,
    std::vector<HloInstruction*>* preset_operands) {
  std::vector<HloInstruction*> operands;
  if (preset_operands) {
    operands = *preset_operands;
  }
  const auto maybe_infer_shape =
      [&](absl::FunctionRef<absl::StatusOr<Shape>()> infer) {
        if (shape.has_value()) {
          return true;
        }
        auto inferred = infer();
        if (!inferred.ok()) {
          return TokenError(
              StrFormat("failed to infer shape for opcode: %s, error: %s",
                        HloOpcodeString(opcode), inferred.status().message()));
        }
        shape = std::move(inferred).value();
        return true;
      };
  const auto create_unary_instruction = [&]() -> HloInstruction* {
    if ((!preset_operands &&
         !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
        !ParseAttributes(attrs, allow_attributes, shape)) {
      return nullptr;
    }
    if (!maybe_infer_shape([&] {
          return ShapeInference::InferUnaryOpShape(opcode, operands[0]);
        })) {
      return nullptr;
    }
    return builder->AddInstruction(
        HloInstruction::CreateUnary(*shape, opcode, operands[0]));
  };
  const auto create_unary_instruction_with_result_accuracy =
      [&]() -> HloInstruction* {
    optional<ResultAccuracy> result_accuracy;
    attrs["result_accuracy"] = {/*required=*/false, AttrTy::kResultAccuracy,
                                &result_accuracy};
    if ((!preset_operands &&
         !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
        !ParseAttributes(attrs, allow_attributes, shape)) {
      return nullptr;
    }
    if (!maybe_infer_shape([&] {
          return ShapeInference::InferUnaryOpShape(opcode, operands[0]);
        })) {
      return nullptr;
    }
    ResultAccuracy accuracy;
    // If the result accuracy is not specified, set it to DEFAULT.
    if (result_accuracy) {
      accuracy = *result_accuracy;
    } else {
      accuracy.set_mode(ResultAccuracy::DEFAULT);
    }
    return builder->AddInstruction(
        HloInstruction::CreateUnary(*shape, opcode, operands[0], accuracy));
  };

  switch (opcode) {
    case HloOpcode::kParameter: {
      int64_t parameter_number;
      if (!ParseToken(TokKind::kLparen,
                      "expects '(' before parameter number") ||
          !ParseInt64(&parameter_number)) {
        return nullptr;
      }
      const LocTy loc = lexer_.GetLoc();
      if (parameter_number < 0) {
        Error(loc, "parameter number must be >= 0");
        return nullptr;
      }
      if (!ParseToken(TokKind::kRparen, "expects ')' after parameter number") ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      std::string param_name(name);
      auto result = builder->AddParameter(HloInstruction::CreateParameter(
          parameter_number, *shape, param_name));
      if (!result.ok()) {
        Error(loc, result.status().message());
        return nullptr;
      }
      return result.value();
    }
    case HloOpcode::kConstant: {
      Literal literal;
      if (!ParseToken(TokKind::kLparen,
                      "expects '(' before constant literal") ||
          !ParseLiteral(&literal, *shape) ||
          !ParseToken(TokKind::kRparen, "expects ')' after constant literal") ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(
          HloInstruction::CreateConstant(std::move(literal)));
    }
    case HloOpcode::kIota: {
      optional<int64_t> iota_dimension;
      attrs["iota_dimension"] = {/*required=*/true, AttrTy::kInt64,
                                 &iota_dimension};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/0)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(
          HloInstruction::CreateIota(*shape, *iota_dimension));
    }
    case HloOpcode::kTopK: {
      optional<int64_t> k;
      attrs["k"] = {/*required=*/true, AttrTy::kInt64, &k};
      optional<bool> largest;
      attrs["largest"] = {/*required=*/false, AttrTy::kBool, &largest};
      if ((!preset_operands && !ParseOperands(&operands, builder,
                                              /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferTopKShape(operands[0]->shape(), *k);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateTopK(
          *shape, operands[0], *k, (largest.has_value() ? *largest : true)));
    }
    // Unary ops with result accuracy.
    case HloOpcode::kExpm1:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kLogistic:
    case HloOpcode::kSqrt:
    case HloOpcode::kCbrt:
    case HloOpcode::kRsqrt:
    case HloOpcode::kTanh:
    case HloOpcode::kErf:
    case HloOpcode::kSin:
    case HloOpcode::kCos:
    case HloOpcode::kTan:
    case HloOpcode::kExp: {
      return create_unary_instruction_with_result_accuracy();
    }
    // Unary ops.
    case HloOpcode::kAbs:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kCopy:
    case HloOpcode::kCopyDone:
    case HloOpcode::kOptimizationBarrier:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kFloor:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kPopulationCount:
    case HloOpcode::kReal:
    case HloOpcode::kSign: {
      return create_unary_instruction();
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
    case HloOpcode::kShiftRightLogical:
    case HloOpcode::kStochasticConvert: {
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/2)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferBinaryOpShape(opcode, operands[0],
                                                      operands[1]);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateBinary(
          *shape, opcode, operands[0], operands[1]));
    }
    // Ternary ops.
    case HloOpcode::kClamp:
    case HloOpcode::kSelect: {
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/3)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferTernaryOpShape(
                opcode, operands[0], operands[1], operands[2]);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateTernary(
          *shape, opcode, operands[0], operands[1], operands[2]));
    }
    // Other supported ops.
    case HloOpcode::kConvert: {
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(
          HloInstruction::CreateConvert(*shape, operands[0]));
    }
    case HloOpcode::kBitcastConvert: {
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(
          HloInstruction::CreateBitcastConvert(*shape, operands[0]));
    }
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart: {
      CollectiveDeviceList device_list;
      optional<int64_t> channel_id;
      optional<std::vector<int64_t>> dimensions;
      optional<bool> constrain_layout;
      optional<bool> use_global_device_ids;
      attrs["replica_groups"] = {/*required=*/false,
                                 AttrTy::kCollectiveDeviceList, &device_list};
      attrs["channel_id"] = {/*required=*/false, AttrTy::kInt64, &channel_id};
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions};
      attrs["constrain_layout"] = {/*required=*/false, AttrTy::kBool,
                                   &constrain_layout};
      attrs["use_global_device_ids"] = {/*required=*/false, AttrTy::kBool,
                                        &use_global_device_ids};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (opcode == HloOpcode::kAllGather) {
        return builder->AddInstruction(HloInstruction::CreateAllGather(
            *shape, operands, dimensions->at(0), device_list,
            constrain_layout ? *constrain_layout : false, channel_id,
            use_global_device_ids ? *use_global_device_ids : false));
      }
      return builder->AddInstruction(HloInstruction::CreateAllGatherStart(
          *shape, operands, dimensions->at(0), device_list,
          constrain_layout ? *constrain_layout : false, channel_id,
          use_global_device_ids ? *use_global_device_ids : false));
    }
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kReduceScatter: {
      CollectiveDeviceList device_list;
      optional<HloComputation*> to_apply;
      optional<int64_t> channel_id;
      optional<bool> constrain_layout;
      optional<bool> use_global_device_ids;
      optional<std::vector<int64_t>> dimensions;
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &to_apply};
      attrs["replica_groups"] = {/*required=*/false,
                                 AttrTy::kCollectiveDeviceList, &device_list};
      attrs["channel_id"] = {/*required=*/false, AttrTy::kInt64, &channel_id};
      attrs["constrain_layout"] = {/*required=*/false, AttrTy::kBool,
                                   &constrain_layout};
      attrs["use_global_device_ids"] = {/*required=*/false, AttrTy::kBool,
                                        &use_global_device_ids};
      if (opcode == HloOpcode::kReduceScatter) {
        attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                               &dimensions};
      }
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (opcode == HloOpcode::kAllReduce) {
        return builder->AddInstruction(HloInstruction::CreateAllReduce(
            *shape, operands, *to_apply, device_list,
            constrain_layout ? *constrain_layout : false, channel_id,
            use_global_device_ids ? *use_global_device_ids : false));
      } else if (opcode == HloOpcode::kReduceScatter) {
        return builder->AddInstruction(HloInstruction::CreateReduceScatter(
            *shape, operands, *to_apply, device_list,
            constrain_layout ? *constrain_layout : false, channel_id,
            use_global_device_ids ? *use_global_device_ids : false,
            dimensions->at(0)));
      }
      return builder->AddInstruction(HloInstruction::CreateAllReduceStart(
          *shape, operands, *to_apply, device_list,
          constrain_layout ? *constrain_layout : false, channel_id,
          use_global_device_ids ? *use_global_device_ids : false));
    }
    case HloOpcode::kAllToAll: {
      CollectiveDeviceList device_list;
      attrs["replica_groups"] = {/*required=*/false,
                                 AttrTy::kCollectiveDeviceList, &device_list};
      optional<int64_t> channel_id;
      attrs["channel_id"] = {/*required=*/false, AttrTy::kInt64, &channel_id};
      optional<std::vector<int64_t>> dimensions;
      attrs["dimensions"] = {/*required=*/false, AttrTy::kBracedInt64List,
                             &dimensions};
      optional<bool> constrain_layout;
      attrs["constrain_layout"] = {/*required=*/false, AttrTy::kBool,
                                   &constrain_layout};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape) ||
          (dimensions && dimensions->size() != 1)) {
        return nullptr;
      }
      optional<int64_t> split_dimension;
      if (dimensions) {
        split_dimension = dimensions->at(0);
      }
      return builder->AddInstruction(HloInstruction::CreateAllToAll(
          *shape, operands, device_list,
          constrain_layout ? *constrain_layout : false, channel_id,
          split_dimension));
    }
    case HloOpcode::kRaggedAllToAll: {
      CollectiveDeviceList device_list;
      attrs["replica_groups"] = {/*required=*/false,
                                 AttrTy::kCollectiveDeviceList, &device_list};
      optional<int64_t> channel_id;
      attrs["channel_id"] = {/*required=*/false, AttrTy::kInt64, &channel_id};
      optional<std::vector<int64_t>> dimensions;
      attrs["dimensions"] = {/*required=*/false, AttrTy::kBracedInt64List,
                             &dimensions};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape) ||
          (dimensions && dimensions->size() != 1)) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateRaggedAllToAll(
          *shape, operands, device_list, channel_id));
    }
    case HloOpcode::kCollectiveBroadcast: {
      CollectiveDeviceList device_list;
      attrs["replica_groups"] = {/*required=*/true,
                                 AttrTy::kCollectiveDeviceList, &device_list};
      optional<int64_t> channel_id;
      attrs["channel_id"] = {/*required=*/false, AttrTy::kInt64, &channel_id};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateCollectiveBroadcast(
          *shape, operands, device_list, false, channel_id));
    }
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart: {
      optional<std::vector<std::vector<int64_t>>> source_targets;
      attrs["source_target_pairs"] = {
          /*required=*/true, AttrTy::kBracedInt64ListList, &source_targets};
      optional<int64_t> channel_id;
      attrs["channel_id"] = {/*required=*/false, AttrTy::kInt64, &channel_id};
      optional<std::vector<std::vector<int64_t>>> slice_sizes;
      attrs["slice_sizes"] = {/*required=*/false, AttrTy::kBracedInt64ListList,
                              &slice_sizes};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      std::vector<std::pair<int64_t, int64_t>> pairs(source_targets->size());
      for (int i = 0; i < pairs.size(); i++) {
        if ((*source_targets)[i].size() != 2) {
          TokenError("expects 'source_target_pairs=' to be a list of pairs");
          return nullptr;
        }
        pairs[i].first = (*source_targets)[i][0];
        pairs[i].second = (*source_targets)[i][1];
      }
      if (!slice_sizes.has_value()) {
        if (opcode == HloOpcode::kCollectivePermute) {
          return builder->AddInstruction(
              HloInstruction::CreateCollectivePermute(*shape, operands, pairs,
                                                      channel_id));
        }
        if (opcode == HloOpcode::kCollectivePermuteStart) {
          return builder->AddInstruction(
              HloInstruction::CreateCollectivePermuteStart(*shape, operands,
                                                           pairs, channel_id));
        }
        LOG(FATAL) << "Expect opcode to be CollectivePermute or "
                      "CollectivePermuteStart, but got "
                   << opcode;
      }
      // TODO update the interface and legalization below for combined
      // collective permutes
      if (operands.size() != 4) {
        TokenError(
            "CollectivePermute and CollectivePermuteStart must "
            "have exactly four operands for dynamic-slice and "
            "in-place update.");
        return nullptr;
      }
      if (opcode == HloOpcode::kCollectivePermute) {
        return builder->AddInstruction(HloInstruction::CreateCollectivePermute(
            *shape, operands[0], operands[1], operands[2], operands[3], pairs,
            *slice_sizes, channel_id));
      }
      if (opcode == HloOpcode::kCollectivePermuteStart) {
        return builder->AddInstruction(
            HloInstruction::CreateCollectivePermuteStart(
                *shape, operands[0], operands[1], operands[2], operands[3],
                pairs, *slice_sizes, channel_id));
      }
      LOG(FATAL) << "Expect opcode to be CollectivePermute or "
                    "CollectivePermuteStart, but got "
                 << opcode;
    }
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone: {
      std::optional<HloComputation*> async_computation;
      if (!preset_operands && !ParseOperands(&operands, builder)) {
        return nullptr;
      }
      auto is_async_shape_correct = [](const Shape& shape) {
        return shape.IsTuple() && shape.tuple_shapes_size() >= 2 &&
               shape.tuple_shapes(0).IsTuple();
      };
      // Verify operand/resulting shapes
      if (opcode == HloOpcode::kAsyncUpdate ||
          opcode == HloOpcode::kAsyncDone) {
        if (operands.size() != 1 ||
            !is_async_shape_correct(operands[0]->shape())) {
          TokenError(
              "AsyncUpdate and AsyncDone expect a single operand in the form "
              "of ((async-operands), async-outputs, state).");
          return nullptr;
        }
      }
      if (opcode == HloOpcode::kAsyncStart ||
          opcode == HloOpcode::kAsyncUpdate) {
        if (!is_async_shape_correct(*shape)) {
          TokenError(
              "AsyncStart and AsyncUpdate expect the op shape to be in the "
              "form of "
              "((async-operands), async-outputs, state).");
          return nullptr;
        }
      }
      // async-{update,done} expect their one singular operand to be the
      // previous async op.
      if (opcode == HloOpcode::kAsyncUpdate ||
          opcode == HloOpcode::kAsyncDone) {
        if (operands.size() != 1 ||
            !is_async_shape_correct(operands[0]->shape())) {
          TokenError(
              "AsyncUpdate and AsyncDone expect a single operand in the form "
              "of ((async-operands), async-outputs, state).");
          return nullptr;
        }
        if (!operands[0]->IsAsynchronous()) {
          TokenError(
              "AsyncUpdate and AsyncDone expect their operand to be the "
              "previous async op.");
          return nullptr;
        }
      }
      optional<std::string> async_execution_thread;
      attrs["async_execution_thread"] = {/*required=*/false, AttrTy::kString,
                                         &async_execution_thread};
      if (async_wrapped_opcode) {
        // Only generate async-wrapper for async-start.
        if (opcode == HloOpcode::kAsyncStart) {
          std::vector<HloInstruction*> async_wrapped_operands;
          std::vector<Shape> async_wrapped_operand_shapes;
          Shape async_wrapped_root_shape;
          async_wrapped_operand_shapes.reserve(operands.size());
          for (const HloInstruction* operand : operands) {
            async_wrapped_operand_shapes.push_back(operand->shape());
          }
          async_wrapped_root_shape = shape->tuple_shapes(1);
          HloComputation::Builder async_wrapped_builder("async_wrapped");
          async_wrapped_operands.reserve(async_wrapped_operand_shapes.size());
          for (int i = 0; i < async_wrapped_operand_shapes.size(); ++i) {
            async_wrapped_operands.push_back(
                async_wrapped_builder.AddInstruction(
                    HloInstruction::CreateParameter(
                        i, async_wrapped_operand_shapes.at(i), "async_param")));
          }
          HloInstruction* root =
              CreateInstruction(&async_wrapped_builder, "async_op",
                                async_wrapped_root_shape, *async_wrapped_opcode,
                                /*async_wrapped_opcode=*/std::nullopt, attrs,
                                allow_attributes, &async_wrapped_operands);
          if (!root) {
            return nullptr;
          }
          computations_.emplace_back(async_wrapped_builder.Build(root));
          async_computation = computations_.back().get();
        } else {
          // Since async-{update,done} will inherit the computation from
          // async-start, we'll only need to make sure it matches what was
          // specified explicitly.
          if (operands[0]->async_wrapped_opcode() != *async_wrapped_opcode) {
            TokenError(
                StrFormat("Expect async wrapped opcode to be %s, but got %s",
                          HloOpcodeString(operands[0]->async_wrapped_opcode()),
                          HloOpcodeString(*async_wrapped_opcode)));
            return nullptr;
          }
        }
      } else {
        attrs["calls"] = {/*required=*/opcode == HloOpcode::kAsyncStart,
                          AttrTy::kHloComputation, &async_computation};
      }
      // Attributes would have already been consumed when constructing the
      // async wrapped computation for async-start.
      if (!(async_wrapped_opcode && opcode == HloOpcode::kAsyncStart)) {
        if (!ParseAttributes(attrs, allow_attributes, shape)) {
          return nullptr;
        }
      }
      // Async attributes on async-{update,done} are allowed for backward
      // compatibility reasons, but are ignored, since they are inherited
      // from the async-start op. Simply check that whatever is explicitly
      // specified matches what is inherited.
      if (opcode == HloOpcode::kAsyncUpdate ||
          opcode == HloOpcode::kAsyncDone) {
        if (async_execution_thread &&
            operands[0]->async_execution_thread() != *async_execution_thread) {
          TokenError(StrFormat(
              "Expect async_execution_thread to be %s, but got %s",
              operands[0]->async_execution_thread(), *async_execution_thread));
          return nullptr;
        }
        if (async_computation &&
            operands[0]->async_wrapped_computation() != *async_computation) {
          TokenError(
              StrFormat("Expect async_wrapped_computation to be %s, but got %s",
                        operands[0]->async_wrapped_computation()->name(),
                        (*async_computation)->name()));
          return nullptr;
        }
      }
      // There should be a 1:1 correspondence between async-start ops and
      // async wrapped computations. At this stage, the computation should
      // not be referenced by any other async op.
      if (opcode == HloOpcode::kAsyncStart &&
          (*async_computation)->IsAsyncComputation()) {
        TokenError(StrFormat(
            "Computation %s is already referenced by another async op",
            (*async_computation)->name()));
        return nullptr;
      }
      if (opcode == HloOpcode::kAsyncStart) {
        // async_execution_thread only needs to be populated for async-start,
        // as the rest of the async chain will reference the root op.
        if (!async_execution_thread) {
          async_execution_thread = HloInstruction::kMainExecutionThread;
        }
        return builder->AddInstruction(HloInstruction::CreateAsyncStart(
            *shape, operands, *async_computation, *async_execution_thread));
      }
      if (opcode == HloOpcode::kAsyncUpdate) {
        return builder->AddInstruction(
            HloInstruction::CreateAsyncUpdate(*shape, operands[0]));
      }
      return builder->AddInstruction(
          HloInstruction::CreateAsyncDone(*shape, operands[0]));
    }
    case HloOpcode::kCopyStart: {
      optional<int> cross_program_prefetch_index = std::nullopt;
      attrs["cross_program_prefetch_index"] = {
          /*required=*/false, AttrTy::kInt32, &cross_program_prefetch_index};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateCopyStart(
          *shape, operands[0], cross_program_prefetch_index));
    }
    case HloOpcode::kReplicaId: {
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/0)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (shape.has_value()) {
        return builder->AddInstruction(HloInstruction::CreateReplicaId(*shape));
      }
      return builder->AddInstruction(HloInstruction::CreateReplicaId());
    }
    case HloOpcode::kPartitionId: {
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/0)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (shape.has_value()) {
        return builder->AddInstruction(
            HloInstruction::CreatePartitionId(*shape));
      }
      return builder->AddInstruction(HloInstruction::CreatePartitionId());
    }
    case HloOpcode::kDynamicReshape: {
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateDynamicReshape(
          *shape, operands[0],
          absl::Span<HloInstruction* const>(operands).subspan(1)));
    }
    case HloOpcode::kReshape: {
      optional<int64_t> inferred_dimension;
      attrs["inferred_dimension"] = {/*required=*/false, AttrTy::kInt64,
                                     &inferred_dimension};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateReshape(
          *shape, operands[0], inferred_dimension.value_or(-1)));
    }
    case HloOpcode::kAfterAll: {
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (operands.empty()) {
        return builder->AddInstruction(HloInstruction::CreateToken());
      }
      return builder->AddInstruction(HloInstruction::CreateAfterAll(operands));
    }
    case HloOpcode::kAddDependency: {
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/2)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(
          HloInstruction::CreateAddDependency(operands[0], operands[1]));
    }
    case HloOpcode::kSort: {
      optional<std::vector<int64_t>> dimensions;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions};
      optional<bool> is_stable = false;
      attrs["is_stable"] = {/*required=*/false, AttrTy::kBool, &is_stable};
      optional<HloComputation*> to_apply;
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &to_apply};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape) ||
          dimensions->size() != 1) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            absl::InlinedVector<const Shape*, 2> arg_shapes;
            arg_shapes.reserve(operands.size());
            for (auto* operand : operands) {
              arg_shapes.push_back(&operand->shape());
            }
            return ShapeInference::InferVariadicOpShape(opcode, arg_shapes);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(
          HloInstruction::CreateSort(*shape, dimensions->at(0), operands,
                                     to_apply.value(), is_stable.value()));
    }
    case HloOpcode::kTuple: {
      if ((!preset_operands &&
           !(shape.has_value()
                 ? ParseOperands(&operands, builder, shape->tuple_shapes_size())
                 : ParseOperands(&operands, builder))) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            absl::InlinedVector<const Shape*, 2> arg_shapes;
            arg_shapes.reserve(operands.size());
            for (auto* operand : operands) {
              arg_shapes.push_back(&operand->shape());
            }
            return ShapeInference::InferVariadicOpShape(opcode, arg_shapes);
          })) {
        return nullptr;
      }
      // HloInstruction::CreateTuple() infers the shape of the tuple from
      // operands and should not be used here.
      return builder->AddInstruction(
          HloInstruction::CreateVariadic(*shape, HloOpcode::kTuple, operands));
    }
    case HloOpcode::kWhile: {
      optional<HloComputation*> condition;
      optional<HloComputation*> body;
      attrs["condition"] = {/*required=*/true, AttrTy::kHloComputation,
                            &condition};
      attrs["body"] = {/*required=*/true, AttrTy::kHloComputation, &body};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferWhileShape(
                condition.value()->ComputeProgramShape(),
                body.value()->ComputeProgramShape(), operands[0]->shape());
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateWhile(
          *shape, *condition, *body, /*init=*/operands[0]));
    }
    case HloOpcode::kRecv: {
      optional<int64_t> channel_id;
      // If the is_host_transfer attribute is not present then default to false.
      optional<bool> is_host_transfer = false;
      attrs["channel_id"] = {/*required=*/false, AttrTy::kInt64, &channel_id};
      attrs["is_host_transfer"] = {/*required=*/false, AttrTy::kBool,
                                   &is_host_transfer};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      // If the is_host_transfer attribute is not present then default to false.
      return builder->AddInstruction(HloInstruction::CreateRecv(
          shape->tuple_shapes(0), operands[0], channel_id, *is_host_transfer));
    }
    case HloOpcode::kRecvDone: {
      optional<int64_t> channel_id;
      // If the is_host_transfer attribute is not present then default to false.
      optional<bool> is_host_transfer = false;
      attrs["channel_id"] = {/*required=*/false, AttrTy::kInt64, &channel_id};
      attrs["is_host_transfer"] = {/*required=*/false, AttrTy::kBool,
                                   &is_host_transfer};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }

      if (dynamic_cast<const HloChannelInstruction*>(operands[0]) != nullptr) {
        if (channel_id != operands[0]->channel_id()) {
          return nullptr;
        }
      }

      return builder->AddInstruction(HloInstruction::CreateRecvDone(
          operands[0], channel_id, *is_host_transfer));
    }
    case HloOpcode::kSend: {
      optional<int64_t> channel_id;
      // If the is_host_transfer attribute is not present then default to false.
      optional<bool> is_host_transfer = false;
      attrs["channel_id"] = {/*required=*/false, AttrTy::kInt64, &channel_id};
      attrs["is_host_transfer"] = {/*required=*/false, AttrTy::kBool,
                                   &is_host_transfer};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/2)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateSend(
          operands[0], operands[1], channel_id, *is_host_transfer));
    }
    case HloOpcode::kSendDone: {
      optional<int64_t> channel_id;
      // If the is_host_transfer attribute is not present then default to false.
      optional<bool> is_host_transfer = false;
      attrs["channel_id"] = {/*required=*/false, AttrTy::kInt64, &channel_id};
      attrs["is_host_transfer"] = {/*required=*/false, AttrTy::kBool,
                                   &is_host_transfer};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }

      if (dynamic_cast<const HloChannelInstruction*>(operands[0]) != nullptr) {
        if (channel_id != operands[0]->channel_id()) {
          return nullptr;
        }
      }

      return builder->AddInstruction(HloInstruction::CreateSendDone(
          operands[0], channel_id, *is_host_transfer));
    }
    case HloOpcode::kGetTupleElement: {
      optional<int64_t> index;
      attrs["index"] = {/*required=*/true, AttrTy::kInt64, &index};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeUtil::GetTupleElementShape(operands[0]->shape(),
                                                   *index);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(
          HloInstruction::CreateGetTupleElement(*shape, operands[0], *index));
    }
    case HloOpcode::kCall: {
      optional<HloComputation*> to_apply;
      optional<bool> is_composite = false;
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &to_apply};
      attrs["is_composite"] = {/*required=*/false, AttrTy::kBool,
                               &is_composite};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            absl::InlinedVector<const Shape*, 2> arg_shapes;
            arg_shapes.reserve(operands.size());
            for (auto* operand : operands) {
              arg_shapes.push_back(&operand->shape());
            }
            return ShapeInference::InferCallShape(
                arg_shapes, to_apply.value()->ComputeProgramShape());
          })) {
        return nullptr;
      }

      auto call_op = HloInstruction::CreateCall(*shape, operands, *to_apply);
      call_op->set_is_composite(is_composite.value());
      return builder->AddInstruction(std::move(call_op));
    }
    case HloOpcode::kReduceWindow: {
      optional<HloComputation*> reduce_computation;
      optional<Window> window;
      attrs["window"] = {/*required=*/false, AttrTy::kWindow, &window};
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &reduce_computation};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!window) {
        window.emplace();
      }
      if (operands.size() % 2) {
        TokenError(StrCat("expects an even number of operands, but has ",
                          operands.size(), " operands"));
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferReduceWindowShape(
                operands[0]->shape(), operands[1]->shape(), *window,
                reduce_computation.value()->ComputeProgramShape());
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateReduceWindow(
          *shape, /*operands=*/
          absl::Span<HloInstruction* const>(operands).subspan(
              0, operands.size() / 2),
          /*init_values=*/
          absl::Span<HloInstruction* const>(operands).subspan(operands.size() /
                                                              2),
          *window, *reduce_computation));
    }
    case HloOpcode::kConvolution: {
      optional<Window> window;
      optional<ConvolutionDimensionNumbers> dnums;
      optional<int64_t> feature_group_count;
      optional<int64_t> batch_group_count;
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
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/2)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
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
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferConvolveShape(
                operands[0]->shape(), operands[1]->shape(),
                *feature_group_count, *batch_group_count, *window, *dnums,
                /*preferred_element_type=*/std::nullopt);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateConvolve(
          *shape, /*lhs=*/operands[0], /*rhs=*/operands[1],
          feature_group_count.value(), batch_group_count.value(), *window,
          *dnums, precision_config));
    }
    case HloOpcode::kFft: {
      optional<FftType> fft_type;
      optional<std::vector<int64_t>> fft_length;
      attrs["fft_type"] = {/*required=*/true, AttrTy::kFftType, &fft_type};
      attrs["fft_length"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &fft_length};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferFftShape(operands[0]->shape(),
                                                 *fft_type, *fft_length);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateFft(
          *shape, operands[0], *fft_type, *fft_length));
    }
    case HloOpcode::kTriangularSolve: {
      TriangularSolveOptions options;
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/2)) ||
          (allow_attributes && !ParseAttributesAsProtoMessage(
                                   /*non_proto_attrs=*/attrs, &options))) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferTriangularSolveShape(
                operands[0]->shape(), operands[1]->shape(), options);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateTriangularSolve(
          *shape, operands[0], operands[1], options));
    }
    case HloOpcode::kCompare: {
      optional<ComparisonDirection> direction;
      optional<Comparison::Type> type;
      attrs["direction"] = {/*required=*/true, AttrTy::kComparisonDirection,
                            &direction};
      attrs["type"] = {/*required=*/false, AttrTy::kComparisonType, &type};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/2)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferBinaryOpShape(opcode, operands[0],
                                                      operands[1]);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateCompare(
          *shape, operands[0], operands[1], *direction, type));
    }
    case HloOpcode::kCholesky: {
      CholeskyOptions options;
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          (allow_attributes && !ParseAttributesAsProtoMessage(
                                   /*non_proto_attrs=*/attrs, &options))) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferCholeskyShape(operands[0]->shape());
          })) {
        return nullptr;
      }
      return builder->AddInstruction(
          HloInstruction::CreateCholesky(*shape, operands[0], options));
    }
    case HloOpcode::kBroadcast: {
      if (!preset_operands &&
          !ParseOperands(&operands, builder, /*expected_size=*/1)) {
        return nullptr;
      }

      // The `dimensions` attr is optional if the broadcasted operand is a
      // scalar; in that case we can infer it to be {}.
      bool operand_is_scalar = ShapeUtil::IsScalar(operands[0]->shape());
      optional<std::vector<int64_t>> broadcast_dimensions;
      attrs["dimensions"] = {/*required=*/!operand_is_scalar,
                             AttrTy::kBracedInt64List, &broadcast_dimensions};
      if (!ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (operand_is_scalar && !broadcast_dimensions.has_value()) {
        broadcast_dimensions.emplace();
      }

      if (!maybe_infer_shape([&] {
            return ShapeInference::InferBroadcastShape(operands[0]->shape(),
                                                       *broadcast_dimensions);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateBroadcast(
          *shape, operands[0], *broadcast_dimensions));
    }
    case HloOpcode::kConcatenate: {
      optional<std::vector<int64_t>> dimensions;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape) ||
          dimensions->size() != 1) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            absl::InlinedVector<const Shape*, 2> arg_shapes;
            arg_shapes.reserve(operands.size());
            for (auto* operand : operands) {
              arg_shapes.push_back(&operand->shape());
            }
            return ShapeInference::InferConcatOpShape(arg_shapes,
                                                      dimensions->at(0));
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateConcatenate(
          *shape, operands, dimensions->at(0)));
    }
    case HloOpcode::kMap: {
      optional<HloComputation*> to_apply;
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &to_apply};
      optional<std::vector<int64_t>> dimensions;
      attrs["dimensions"] = {/*required=*/false, AttrTy::kBracedInt64List,
                             &dimensions};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            absl::InlinedVector<const Shape*, 2> arg_shapes;
            arg_shapes.reserve(operands.size());
            for (auto* operand : operands) {
              arg_shapes.push_back(&operand->shape());
            }
            return ShapeInference::InferMapShape(
                arg_shapes, to_apply.value()->ComputeProgramShape(),
                *dimensions);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(
          HloInstruction::CreateMap(*shape, operands, *to_apply));
    }
    case HloOpcode::kReduce: {
      optional<HloComputation*> reduce_computation;
      attrs["to_apply"] = {/*required=*/true, AttrTy::kHloComputation,
                           &reduce_computation};
      optional<std::vector<int64_t>> dimensions_to_reduce;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions_to_reduce};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (operands.size() % 2) {
        TokenError(StrCat("expects an even number of operands, but has ",
                          operands.size(), " operands"));
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            absl::InlinedVector<const Shape*, 2> arg_shapes;
            arg_shapes.reserve(operands.size());
            for (auto* operand : operands) {
              arg_shapes.push_back(&operand->shape());
            }
            return ShapeInference::InferReduceShape(
                arg_shapes, *dimensions_to_reduce,
                reduce_computation.value()->ComputeProgramShape());
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateReduce(
          *shape, /*operands=*/
          absl::Span<HloInstruction* const>(operands).subspan(
              0, operands.size() / 2),
          /*init_values=*/
          absl::Span<HloInstruction* const>(operands).subspan(operands.size() /
                                                              2),
          *dimensions_to_reduce, *reduce_computation));
    }
    case HloOpcode::kReverse: {
      optional<std::vector<int64_t>> dimensions;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferReverseShape(operands[0]->shape(),
                                                     *dimensions);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(
          HloInstruction::CreateReverse(*shape, operands[0], *dimensions));
    }
    case HloOpcode::kSelectAndScatter: {
      optional<HloComputation*> select;
      attrs["select"] = {/*required=*/true, AttrTy::kHloComputation, &select};
      optional<HloComputation*> scatter;
      attrs["scatter"] = {/*required=*/true, AttrTy::kHloComputation, &scatter};
      optional<Window> window;
      attrs["window"] = {/*required=*/false, AttrTy::kWindow, &window};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/3)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!window) {
        window.emplace();
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferSelectAndScatterShape(
                operands[0]->shape(), select.value()->ComputeProgramShape(),
                *window, operands[1]->shape(), operands[2]->shape(),
                scatter.value()->ComputeProgramShape());
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateSelectAndScatter(
          *shape, /*operand=*/operands[0], *select, *window,
          /*source=*/operands[1], /*init_value=*/operands[2], *scatter));
    }
    case HloOpcode::kSlice: {
      optional<SliceRanges> slice_ranges;
      attrs["slice"] = {/*required=*/true, AttrTy::kSliceRanges, &slice_ranges};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateSlice(
          *shape, operands[0], slice_ranges->starts, slice_ranges->limits,
          slice_ranges->strides));
    }
    case HloOpcode::kDynamicSlice: {
      optional<std::vector<int64_t>> dynamic_slice_sizes;
      attrs["dynamic_slice_sizes"] = {
          /*required=*/true, AttrTy::kBracedInt64List, &dynamic_slice_sizes};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (operands.empty()) {
        TokenError("Expected at least one operand.");
        return nullptr;
      }
      if (!(operands.size() == 2 &&
            operands[1]->shape().dimensions().size() == 1) &&
          operands.size() != 1 + operands[0]->shape().dimensions().size()) {
        TokenError("Wrong number of operands.");
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateDynamicSlice(
          *shape, /*operand=*/operands[0],
          /*start_indices=*/absl::MakeSpan(operands).subspan(1),
          *dynamic_slice_sizes));
    }
    case HloOpcode::kDynamicUpdateSlice: {
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (operands.size() < 2) {
        TokenError("Expected at least two operands.");
        return nullptr;
      }
      if (!(operands.size() == 3 &&
            operands[2]->shape().dimensions().size() == 1) &&
          operands.size() != 2 + operands[0]->shape().dimensions().size()) {
        TokenError("Wrong number of operands.");
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
          *shape, /*operand=*/operands[0], /*update=*/operands[1],
          /*start_indices=*/absl::MakeSpan(operands).subspan(2)));
    }
    case HloOpcode::kTranspose: {
      optional<std::vector<int64_t>> dimensions;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferTransposeShape(operands[0]->shape(),
                                                       *dimensions);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(
          HloInstruction::CreateTranspose(*shape, operands[0], *dimensions));
    }
    case HloOpcode::kBatchNormTraining: {
      optional<float> epsilon;
      attrs["epsilon"] = {/*required=*/true, AttrTy::kFloat, &epsilon};
      optional<int64_t> feature_index;
      attrs["feature_index"] = {/*required=*/true, AttrTy::kInt64,
                                &feature_index};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/3)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferBatchNormTrainingShape(
                operands[0]->shape(), operands[1]->shape(),
                operands[2]->shape(), *feature_index);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateBatchNormTraining(
          *shape, /*operand=*/operands[0], /*scale=*/operands[1],
          /*offset=*/operands[2], *epsilon, *feature_index));
    }
    case HloOpcode::kBatchNormInference: {
      optional<float> epsilon;
      attrs["epsilon"] = {/*required=*/true, AttrTy::kFloat, &epsilon};
      optional<int64_t> feature_index;
      attrs["feature_index"] = {/*required=*/true, AttrTy::kInt64,
                                &feature_index};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/5)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferBatchNormInferenceShape(
                operands[0]->shape(), operands[1]->shape(),
                operands[2]->shape(), operands[3]->shape(),
                operands[4]->shape(), *feature_index);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateBatchNormInference(
          *shape, /*operand=*/operands[0], /*scale=*/operands[1],
          /*offset=*/operands[2], /*mean=*/operands[3],
          /*variance=*/operands[4], *epsilon, *feature_index));
    }
    case HloOpcode::kBatchNormGrad: {
      optional<float> epsilon;
      attrs["epsilon"] = {/*required=*/true, AttrTy::kFloat, &epsilon};
      optional<int64_t> feature_index;
      attrs["feature_index"] = {/*required=*/true, AttrTy::kInt64,
                                &feature_index};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/5)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferBatchNormGradShape(
                operands[0]->shape(), operands[1]->shape(),
                operands[2]->shape(), operands[3]->shape(),
                operands[4]->shape(), *feature_index);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateBatchNormGrad(
          *shape, /*operand=*/operands[0], /*scale=*/operands[1],
          /*mean=*/operands[2], /*variance=*/operands[3],
          /*grad_output=*/operands[4], *epsilon, *feature_index));
    }
    case HloOpcode::kPad: {
      optional<PaddingConfig> padding;
      attrs["padding"] = {/*required=*/true, AttrTy::kPaddingConfig, &padding};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/2)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferPadShape(
                operands[0]->shape(), operands[1]->shape(), *padding);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreatePad(
          *shape, operands[0], /*padding_value=*/operands[1], *padding));
    }
    case HloOpcode::kFusion: {
      optional<HloComputation*> fusion_computation;
      attrs["calls"] = {/*required=*/true, AttrTy::kHloComputation,
                        &fusion_computation};
      optional<HloInstruction::FusionKind> fusion_kind;
      attrs["kind"] = {/*required=*/true, AttrTy::kFusionKind, &fusion_kind};
      optional<
          std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>>
          output_to_operand_aliasing;
      attrs["output_to_operand_aliasing"] = {/*required=*/false,
                                             AttrTy::kInstructionAliasing,
                                             &output_to_operand_aliasing};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      auto instr = builder->AddInstruction(HloInstruction::CreateFusion(
          *shape, *fusion_kind, operands, *fusion_computation));
      auto fusion_instr = Cast<HloFusionInstruction>(instr);
      if (output_to_operand_aliasing.has_value()) {
        fusion_instr->set_output_to_operand_aliasing(
            std::move(*output_to_operand_aliasing));
      }
      return instr;
    }
    case HloOpcode::kInfeed: {
      optional<std::string> config;
      attrs["infeed_config"] = {/*required=*/false, AttrTy::kString, &config};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      // We need to know the infeed data shape to construct the infeed
      // instruction. This is the zero-th element of the tuple-shaped output of
      // the infeed instruction. ShapeUtil::GetTupleElementShape will check fail
      // if the shape is not a non-empty tuple, so add guard so an error message
      // can be emitted instead of a check fail
      if (!shape->IsTuple() && !ShapeUtil::IsEmptyTuple(*shape)) {
        TokenError("infeed must have a non-empty tuple shape");
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateInfeed(
          ShapeUtil::GetTupleElementShape(*shape, 0), operands[0],
          config ? *config : ""));
    }
    case HloOpcode::kOutfeed: {
      optional<std::string> config;
      optional<Shape> outfeed_shape;
      attrs["outfeed_config"] = {/*required=*/false, AttrTy::kString, &config};
      attrs["outfeed_shape"] = {/*required=*/false, AttrTy::kShape,
                                &outfeed_shape};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/2)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      HloInstruction* const outfeed_input = operands[0];
      HloInstruction* const outfeed_token = operands[1];
      const Shape shape =
          outfeed_shape.has_value() ? *outfeed_shape : outfeed_input->shape();
      return builder->AddInstruction(HloInstruction::CreateOutfeed(
          shape, outfeed_input, outfeed_token, config ? *config : ""));
    }
    case HloOpcode::kRng: {
      optional<RandomDistribution> distribution;
      attrs["distribution"] = {/*required=*/true, AttrTy::kDistribution,
                               &distribution};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(
          HloInstruction::CreateRng(*shape, *distribution, operands));
    }
    case HloOpcode::kRngGetAndUpdateState: {
      optional<int64_t> delta;
      attrs["delta"] = {/*required=*/true, AttrTy::kInt64, &delta};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/0)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(
          HloInstruction::CreateRngGetAndUpdateState(*shape, *delta));
    }
    case HloOpcode::kRngBitGenerator: {
      optional<RandomAlgorithm> algorithm;
      attrs["algorithm"] = {/*required=*/true, AttrTy::kRandomAlgorithm,
                            &algorithm};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateRngBitGenerator(
          *shape, operands[0], *algorithm));
    }
    case HloOpcode::kReducePrecision: {
      optional<int64_t> exponent_bits;
      optional<int64_t> mantissa_bits;
      attrs["exponent_bits"] = {/*required=*/true, AttrTy::kInt64,
                                &exponent_bits};
      attrs["mantissa_bits"] = {/*required=*/true, AttrTy::kInt64,
                                &mantissa_bits};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateReducePrecision(
          *shape, operands[0], static_cast<int>(*exponent_bits),
          static_cast<int>(*mantissa_bits)));
    }
    case HloOpcode::kConditional: {
      optional<HloComputation*> true_computation;
      optional<HloComputation*> false_computation;
      optional<std::vector<HloComputation*>> branch_computations;
      if (!preset_operands && !ParseOperands(&operands, builder)) {
        return nullptr;
      }
      if (!ShapeUtil::IsScalar(operands[0]->shape())) {
        TokenError("The first operand must be a scalar");
        return nullptr;
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
          TokenError("The first operand must be a scalar of PRED or S32");
          return nullptr;
        }
        attrs["branch_computations"] = {/*required=*/true,
                                        AttrTy::kBracedHloComputationList,
                                        &branch_computations};
      }
      if (!ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (branch_index_is_bool) {
        branch_computations.emplace({*true_computation, *false_computation});
      }
      if (branch_computations->empty() ||
          operands.size() != branch_computations->size() + 1) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            absl::InlinedVector<ProgramShape, 2> branch_computation_shapes;
            branch_computation_shapes.reserve(branch_computations->size());
            for (auto* computation : *branch_computations) {
              branch_computation_shapes.push_back(
                  computation->ComputeProgramShape());
            }
            absl::InlinedVector<Shape, 2> branch_operand_shapes;
            branch_operand_shapes.reserve(operands.size() - 1);
            for (int i = 1; i < operands.size(); ++i) {
              branch_operand_shapes.push_back(operands[i]->shape());
            }
            return ShapeInference::InferConditionalShape(
                operands[0]->shape(), branch_computation_shapes,
                branch_operand_shapes);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateConditional(
          *shape, /*branch_index=*/operands[0],
          absl::MakeSpan(*branch_computations),
          absl::MakeSpan(operands).subspan(1)));
    }
    case HloOpcode::kCustomCall: {
      optional<std::string> custom_call_target;
      optional<Window> window;
      optional<ConvolutionDimensionNumbers> dnums;
      optional<int64_t> feature_group_count;
      optional<int64_t> batch_group_count;
      optional<std::vector<Shape>> operand_layout_constraints;
      optional<bool> custom_call_has_side_effect;
      optional<HloComputation*> to_apply;
      optional<
          std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>>
          output_to_operand_aliasing;
      optional<PaddingType> padding_type;
      optional<std::vector<HloComputation*>> called_computations;
      optional<CustomCallSchedule> custom_call_schedule;
      optional<CustomCallApiVersion> api_version;
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
      attrs["called_computations"] = {/*required=*/false,
                                      AttrTy::kBracedHloComputationList,
                                      &called_computations};
      attrs["output_to_operand_aliasing"] = {/*required=*/false,
                                             AttrTy::kInstructionAliasing,
                                             &output_to_operand_aliasing};

      attrs["padding_type"] = {/*required=*/false, AttrTy::kPaddingType,
                               &padding_type};

      optional<Literal> literal;
      attrs["literal"] = {/*required=*/false, AttrTy::kLiteral, &literal};
      optional<std::vector<PrecisionConfig::Precision>> operand_precision;
      attrs["operand_precision"] = {/*required=*/false, AttrTy::kPrecisionList,
                                    &operand_precision};
      HloInstruction* instruction;
      if (called_computations.has_value() && to_apply.has_value()) {
        TokenError(
            "A single instruction can't have both to_apply and "
            "calls field");
        return nullptr;
      }
      attrs["schedule"] = {/*required=*/false, AttrTy::kCustomCallSchedule,
                           &custom_call_schedule};
      attrs["api_version"] = {/*required=*/false, AttrTy::kCustomCallApiVersion,
                              &api_version};
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }

      if (api_version.has_value() &&
          *api_version == CustomCallApiVersion::API_VERSION_UNSPECIFIED) {
        TokenError(StrCat("Invalid API version: ",
                          CustomCallApiVersion_Name(*api_version)));
        return nullptr;
      }
      if (operand_layout_constraints.has_value()) {
        if (!LayoutUtil::HasLayout(*shape)) {
          TokenError("Layout must be set on layout-constrained custom call");
          return nullptr;
        }
        if (operands.size() != operand_layout_constraints->size()) {
          TokenError(StrCat("Expected ", operands.size(),
                            " operand layout constraints, ",
                            operand_layout_constraints->size(), " given"));
          return nullptr;
        }
        for (int64_t i = 0; i < operands.size(); ++i) {
          const Shape& operand_shape_with_layout =
              (*operand_layout_constraints)[i];
          if (!LayoutUtil::HasLayout(operand_shape_with_layout)) {
            TokenError(StrCat(
                "Operand layout constraint shape ",
                ShapeUtil::HumanStringWithLayout(operand_shape_with_layout),
                " for operand ", i, " does not have a layout"));
            return nullptr;
          }
          if (!ShapeUtil::Compatible(operand_shape_with_layout,
                                     operands[i]->shape())) {
            TokenError(StrCat(
                "Operand layout constraint shape ",
                ShapeUtil::HumanStringWithLayout(operand_shape_with_layout),
                " for operand ", i, " is not compatible with operand shape ",
                ShapeUtil::HumanStringWithLayout(operands[i]->shape())));
            return nullptr;
          }
        }
        instruction = builder->AddInstruction(HloInstruction::CreateCustomCall(
            *shape, operands, *custom_call_target, *operand_layout_constraints,
            ""));
      } else {
        if (to_apply.has_value()) {
          instruction =
              builder->AddInstruction(HloInstruction::CreateCustomCall(
                  *shape, operands, *to_apply, *custom_call_target, ""));
        } else if (called_computations.has_value()) {
          instruction =
              builder->AddInstruction(HloInstruction::CreateCustomCall(
                  *shape, operands, *called_computations, *custom_call_target,
                  ""));
        } else {
          instruction =
              builder->AddInstruction(HloInstruction::CreateCustomCall(
                  *shape, operands, *custom_call_target, ""));
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
      if (padding_type.has_value()) {
        custom_call_instr->set_padding_type(*padding_type);
      }
      if (custom_call_has_side_effect.has_value()) {
        custom_call_instr->set_custom_call_has_side_effect(
            *custom_call_has_side_effect);
      }
      if (custom_call_schedule.has_value()) {
        custom_call_instr->set_custom_call_schedule(*custom_call_schedule);
      }
      if (api_version.has_value()) {
        custom_call_instr->set_api_version(*api_version);
      }
      if (output_to_operand_aliasing.has_value()) {
        custom_call_instr->set_output_to_operand_aliasing(
            std::move(*output_to_operand_aliasing));
      }
      if (literal.has_value()) {
        custom_call_instr->set_literal(std::move(*literal));
      }
      PrecisionConfig precision_config;
      if (operand_precision) {
        *precision_config.mutable_operand_precision() = {
            operand_precision->begin(), operand_precision->end()};
      } else {
        precision_config.mutable_operand_precision()->Resize(
            operands.size(), PrecisionConfig::DEFAULT);
      }
      *custom_call_instr->mutable_precision_config() = precision_config;
      return instruction;
    }
    case HloOpcode::kDot: {
      optional<std::vector<int64_t>> lhs_contracting_dims;
      attrs["lhs_contracting_dims"] = {
          /*required=*/false, AttrTy::kBracedInt64List, &lhs_contracting_dims};
      optional<std::vector<int64_t>> rhs_contracting_dims;
      attrs["rhs_contracting_dims"] = {
          /*required=*/false, AttrTy::kBracedInt64List, &rhs_contracting_dims};
      optional<std::vector<int64_t>> lhs_batch_dims;
      attrs["lhs_batch_dims"] = {/*required=*/false, AttrTy::kBracedInt64List,
                                 &lhs_batch_dims};
      optional<std::vector<int64_t>> rhs_batch_dims;
      attrs["rhs_batch_dims"] = {/*required=*/false, AttrTy::kBracedInt64List,
                                 &rhs_batch_dims};
      optional<std::vector<PrecisionConfig::Precision>> operand_precision;
      attrs["operand_precision"] = {/*required=*/false, AttrTy::kPrecisionList,
                                    &operand_precision};
      std::vector<SparsityDescriptor> sparsity;
      attrs["sparsity"] = {/*required=*/false, AttrTy::kSparsityDescriptor,
                           &sparsity};

      optional<PrecisionConfig::Algorithm> algorithm;
      attrs["algorithm"] = {/*required=*/false, AttrTy::kPrecisionAlgorithm,
                            &algorithm};

      LocTy loc = lexer_.GetLoc();
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }

      int expected_size = HloDotInstruction::kOperands + sparsity.size();
      if (sparsity.size() > HloDotInstruction::kOperands) {
        Error(loc,
              StrCat("too many sparse dot descriptors: ", sparsity.size()));
        return nullptr;
      }
      if (operands.size() != expected_size) {
        Error(loc, StrCat("expects ", expected_size, " operands, but has ",
                          operands.size(), " operands"));
        return nullptr;
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
            HloDotInstruction::kOperands, PrecisionConfig::DEFAULT);
      }
      if (algorithm) {
        precision_config.set_algorithm(*algorithm);
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferDotOpShape(
                operands[0]->shape(), operands[1]->shape(), dnum,
                /*preferred_element_type=*/std::nullopt, sparsity);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateDot(
          *shape, operands[0], operands[1], dnum, precision_config, sparsity,
          absl::MakeSpan(operands).subspan(HloDotInstruction::kOperands)));
    }
    case HloOpcode::kRaggedDot: {
      optional<std::vector<int64_t>> lhs_contracting_dims;
      attrs["lhs_contracting_dims"] = {
          /*required=*/false, AttrTy::kBracedInt64List, &lhs_contracting_dims};
      optional<std::vector<int64_t>> rhs_contracting_dims;
      attrs["rhs_contracting_dims"] = {
          /*required=*/false, AttrTy::kBracedInt64List, &rhs_contracting_dims};
      optional<std::vector<int64_t>> lhs_batch_dims;
      attrs["lhs_batch_dims"] = {/*required=*/false, AttrTy::kBracedInt64List,
                                 &lhs_batch_dims};
      optional<std::vector<int64_t>> rhs_batch_dims;
      attrs["rhs_batch_dims"] = {/*required=*/false, AttrTy::kBracedInt64List,
                                 &rhs_batch_dims};
      optional<std::vector<int64_t>> lhs_ragged_dims;
      attrs["lhs_ragged_dims"] = {/*required=*/false, AttrTy::kBracedInt64List,
                                  &lhs_ragged_dims};
      optional<std::vector<int64_t>> rhs_group_dims;
      attrs["rhs_group_dims"] = {/*required=*/false, AttrTy::kBracedInt64List,
                                 &rhs_group_dims};
      optional<std::vector<PrecisionConfig::Precision>> operand_precision;
      attrs["operand_precision"] = {/*required=*/false, AttrTy::kPrecisionList,
                                    &operand_precision};

      LocTy loc = lexer_.GetLoc();
      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }

      int expected_size = HloRaggedDotInstruction::kOperands;
      if (operands.size() != expected_size) {
        Error(loc, StrCat("expects ", expected_size, " operands, but has ",
                          operands.size(), " operands"));
        return nullptr;
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
      RaggedDotDimensionNumbers ragged_dnum;
      *ragged_dnum.mutable_dot_dimension_numbers() = dnum;
      if (lhs_ragged_dims) {
        *ragged_dnum.mutable_lhs_ragged_dimensions() = {
            lhs_ragged_dims->begin(), lhs_ragged_dims->end()};
      }
      if (rhs_group_dims) {
        *ragged_dnum.mutable_rhs_group_dimensions() = {rhs_group_dims->begin(),
                                                       rhs_group_dims->end()};
      }

      PrecisionConfig precision_config;
      if (operand_precision) {
        *precision_config.mutable_operand_precision() = {
            operand_precision->begin(), operand_precision->end()};
      } else {
        // Only the lhs and rhs operands have precision.
        precision_config.mutable_operand_precision()->Resize(
            HloRaggedDotInstruction::kOperands - 1, PrecisionConfig::DEFAULT);
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferRaggedDotOpShape(
                operands[0]->shape(), operands[1]->shape(),
                operands[2]->shape(), ragged_dnum,
                /*preferred_element_type=*/std::nullopt);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateRaggedDot(
          *shape, operands[0], operands[1], operands[2], ragged_dnum,
          precision_config));
    }
    case HloOpcode::kGather: {
      optional<std::vector<int64_t>> offset_dims;
      attrs["offset_dims"] = {/*required=*/true, AttrTy::kBracedInt64List,
                              &offset_dims};
      optional<std::vector<int64_t>> collapsed_slice_dims;
      attrs["collapsed_slice_dims"] = {
          /*required=*/true, AttrTy::kBracedInt64List, &collapsed_slice_dims};
      optional<std::vector<int64_t>> start_index_map;
      attrs["start_index_map"] = {/*required=*/true, AttrTy::kBracedInt64List,
                                  &start_index_map};
      optional<int64_t> index_vector_dim;
      attrs["index_vector_dim"] = {/*required=*/true, AttrTy::kInt64,
                                   &index_vector_dim};
      optional<std::vector<int64_t>> slice_sizes;
      attrs["slice_sizes"] = {/*required=*/true, AttrTy::kBracedInt64List,
                              &slice_sizes};
      optional<bool> indices_are_sorted = false;
      attrs["indices_are_sorted"] = {/*required=*/false, AttrTy::kBool,
                                     &indices_are_sorted};
      optional<std::vector<int64_t>> operand_batching_dims;
      attrs["operand_batching_dims"] = {
          /*required=*/false, AttrTy::kBracedInt64List, &operand_batching_dims};
      optional<std::vector<int64_t>> start_indices_batching_dims;
      attrs["start_indices_batching_dims"] = {/*required=*/false,
                                              AttrTy::kBracedInt64List,
                                              &start_indices_batching_dims};

      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/2)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }

      GatherDimensionNumbers dim_numbers =
          HloGatherInstruction::MakeGatherDimNumbers(
              /*offset_dims=*/*offset_dims,
              /*collapsed_slice_dims=*/*collapsed_slice_dims,
              /*start_index_map=*/*start_index_map,
              /*index_vector_dim=*/*index_vector_dim,
              /*operand_batching_dims=*/
              operand_batching_dims ? *operand_batching_dims
                                    : std::vector<int64_t>(),
              /*start_indices_batching_dims=*/
              start_indices_batching_dims ? *start_indices_batching_dims
                                          : std::vector<int64_t>());
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferGatherShape(operands[0]->shape(),
                                                    operands[1]->shape(),
                                                    dim_numbers, *slice_sizes);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateGather(
          *shape, /*operand=*/operands[0], /*start_indices=*/operands[1],
          dim_numbers, *slice_sizes, indices_are_sorted.value()));
    }
    case HloOpcode::kScatter: {
      optional<std::vector<int64_t>> update_window_dims;
      attrs["update_window_dims"] = {
          /*required=*/true, AttrTy::kBracedInt64List, &update_window_dims};
      optional<std::vector<int64_t>> inserted_window_dims;
      attrs["inserted_window_dims"] = {
          /*required=*/true, AttrTy::kBracedInt64List, &inserted_window_dims};
      optional<std::vector<int64_t>> scatter_dims_to_operand_dims;
      attrs["scatter_dims_to_operand_dims"] = {/*required=*/true,
                                               AttrTy::kBracedInt64List,
                                               &scatter_dims_to_operand_dims};
      optional<int64_t> index_vector_dim;
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
      optional<std::vector<int64_t>> input_batching_dims;
      attrs["input_batching_dims"] = {
          /*required=*/false, AttrTy::kBracedInt64List, &input_batching_dims};
      optional<std::vector<int64_t>> scatter_indices_batching_dims;
      attrs["scatter_indices_batching_dims"] = {/*required=*/false,
                                                AttrTy::kBracedInt64List,
                                                &scatter_indices_batching_dims};

      if ((!preset_operands && !ParseOperands(&operands, builder)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }

      if (operands.size() % 2 == 0) {
        TokenError(StrCat("expects an odd number of operands, but has ",
                          operands.size(), " operands"));
        return nullptr;
      }

      ScatterDimensionNumbers dim_numbers =
          HloScatterInstruction::MakeScatterDimNumbers(
              /*update_window_dims=*/*update_window_dims,
              /*inserted_window_dims=*/*inserted_window_dims,
              /*scatter_dims_to_operand_dims=*/*scatter_dims_to_operand_dims,
              /*index_vector_dim=*/*index_vector_dim,
              /*input_batching_dims=*/
              input_batching_dims ? *input_batching_dims
                                  : std::vector<int64_t>(),
              /*scatter_indices_batching_dims=*/
              scatter_indices_batching_dims ? *scatter_indices_batching_dims
                                            : std::vector<int64_t>());

      if (!maybe_infer_shape([&] {
            absl::InlinedVector<const Shape*, 3> arg_shapes;
            arg_shapes.reserve(operands.size());
            for (auto* operand : operands) {
              arg_shapes.push_back(&operand->shape());
            }
            return ShapeInference::InferScatterShape(
                arg_shapes, update_computation.value()->ComputeProgramShape(),
                dim_numbers);
          })) {
        return nullptr;
      }
      auto input_count = operands.size() / 2;
      auto operand_span = absl::MakeConstSpan(operands);
      return builder->AddInstruction(HloInstruction::CreateScatter(
          *shape, operand_span.first(input_count), operands[input_count],
          operand_span.last(input_count), *update_computation, dim_numbers,
          indices_are_sorted.value(), unique_indices.value()));
    }
    case HloOpcode::kDomain: {
      DomainData domain;
      attrs["domain"] = {/*required=*/true, AttrTy::kDomain, &domain};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferUnaryOpShape(opcode, operands[0]);
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateDomain(
          *shape, operands[0], std::move(domain.exit_metadata),
          std::move(domain.entry_metadata)));
    }
    case HloOpcode::kGetDimensionSize: {
      optional<std::vector<int64_t>> dimensions;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/1)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferGetDimensionSizeShape(
                operands[0]->shape(), dimensions->at(0));
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateGetDimensionSize(
          *shape, operands[0], (*dimensions)[0]));
    }
    case HloOpcode::kSetDimensionSize: {
      optional<std::vector<int64_t>> dimensions;
      attrs["dimensions"] = {/*required=*/true, AttrTy::kBracedInt64List,
                             &dimensions};
      if ((!preset_operands &&
           !ParseOperands(&operands, builder, /*expected_size=*/2)) ||
          !ParseAttributes(attrs, allow_attributes, shape)) {
        return nullptr;
      }
      if (!maybe_infer_shape([&] {
            return ShapeInference::InferSetDimensionSizeShape(
                operands[0]->shape(), operands[1]->shape(), dimensions->at(0));
          })) {
        return nullptr;
      }
      return builder->AddInstruction(HloInstruction::CreateSetDimensionSize(
          *shape, operands[0], operands[1], (*dimensions)[0]));
    }
    default:
      return nullptr;
  }
}  // NOLINT(readability/fn_size)

// ::= '{' <full_device_list> '}' | iota_list
// full_device_list ::= '{' <int_list> '}' ( ',' '{' <int_list> '}' )*
// iota_list ::= ('[' d ']')  '<=[' reshape_d ']' ('T(' transpose_d ')')?
// d ::= int_list
// reshape_d ::= int_list
// transpose_d ::= int_list
bool HloParserImpl::ParseCollectiveDeviceList(
    CollectiveDeviceList* device_list) {
  // If the first token is a '{', then we are parsing legacy version of
  // collective device list, which is a list of lists.
  if (lexer_.GetKind() == TokKind::kLbrace) {
    std::vector<ReplicaGroup> replica_groups;
    if (!ParseReplicaGroupsOnly(&replica_groups)) {
      return false;
    }
    *device_list = CollectiveDeviceList(replica_groups);
    return true;
  }

  // Otherwise, we are parsing the new version of collective device list, which
  // is an iota tile assignment.
  std::vector<int64_t> tile_assignment_dimensions;
  std::vector<int64_t> iota_reshape_dims;
  std::vector<int> iota_transpose_perm;
  // Parse the tile assignment expecting an iota tile assignment.
  if (!ParseTileAssignment(tile_assignment_dimensions, iota_reshape_dims,
                           iota_transpose_perm, nullptr)) {
    return false;
  }

  // Iota tile assignment associated with collective device list should only
  // have 2 dimensions.
  if (tile_assignment_dimensions.size() != 2) {
    VLOG(kErrorLevel)
        << "Expected tile assignment to have 2 dimensions for collective "
           "device list but got "
        << tile_assignment_dimensions.size();
    return false;
  }

  *device_list = CollectiveDeviceList(IotaReplicaGroupList(
      tile_assignment_dimensions[0], tile_assignment_dimensions[1],
      iota_reshape_dims, iota_transpose_perm));
  return true;
}

// ::= '{' (single_sharding | tuple_sharding) '}'
//
// tuple_sharding ::= single_sharding* (',' single_sharding)*
bool HloParserImpl::ParseSharding(std::optional<HloSharding>& sharding) {
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
  std::vector<HloSharding> tuple_shardings;
  if (lexer_.GetKind() != TokKind::kRbrace) {
    do {
      std::optional<HloSharding> tuple_sharding;
      if (!ParseSingleSharding(tuple_sharding,
                               /*lbrace_pre_lexed=*/false)) {
        return false;
      }
      tuple_shardings.push_back(std::move(*tuple_sharding));
    } while (EatIfPresent(TokKind::kComma));
  }
  sharding = HloSharding::FlatTuple(std::move(tuple_shardings));

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

      std::string result;
      if (lexer_.GetKind() == TokKind::kString) {
        if (!ParseString(&result)) {
          return false;
        }
      } else if (lexer_.GetKind() == TokKind::kLbrace) {
        if (!ParseJsonDict(&result)) {
          return false;
        }
      } else {
        return false;
      }

      (*frontend_attributes->mutable_map())[attribute] = result;
    } while (EatIfPresent(TokKind::kComma));
  }
  return ParseToken(TokKind::kRbrace,
                    "expects '}' at the end of frontend attributes");
}

// statistics
//    ::= '{' /*empty*/ '}'
//    ::= '{' index, single_statistic '}'
// index ::= 'visualizing_index=' value
// single_statistic ::= statistic '=' value (',' statistic '=' value)*
bool HloParserImpl::ParseStatisticsViz(StatisticsViz* statistics_viz) {
  CHECK(statistics_viz != nullptr);
  if (!ParseToken(TokKind::kLbrace, "expected '{' to start statistics")) {
    return false;
  }
  if (lexer_.GetKind() == TokKind::kRbrace) {
    // empty
  } else {
    // index must exist
    std::string visualizing_index_attr_name;
    if (!ParseAttributeName(&visualizing_index_attr_name)) {
      return false;
    }
    if (lexer_.GetKind() != TokKind::kInt) {
      return false;
    }
    statistics_viz->set_stat_index_to_visualize(lexer_.GetInt64Val());
    lexer_.Lex();

    // then process statistics
    while (EatIfPresent(TokKind::kComma)) {
      std::string stat_name;
      if (!ParseAttributeName(&stat_name)) {
        return false;
      }
      if (lexer_.GetKind() != TokKind::kDecimal &&
          lexer_.GetKind() != TokKind::kInt) {
        return false;
      }
      Statistic statistic;
      statistic.set_stat_name(stat_name);
      statistic.set_stat_val(lexer_.GetKind() == TokKind::kDecimal
                                 ? lexer_.GetDecimalVal()
                                 : lexer_.GetInt64Val());
      lexer_.Lex();
      *statistics_viz->add_statistics() = std::move(statistic);
    }
  }
  return ParseToken(TokKind::kRbrace, "expects '}' at the end of statistics");
}

// devices argument is optional: if not present, the tile assignment is assumed
// to be an iota tile assignment.
bool HloParserImpl::ParseTileAssignment(
    std::vector<int64_t>& tile_assignment_dimensions,
    std::vector<int64_t>& iota_reshape_dims,
    std::vector<int>& iota_transpose_perm, std::vector<int64_t>* devices) {
  if (!ParseToken(TokKind::kLsquare,
                  "expected '[' to start sharding devices shape")) {
    return false;
  }

  do {
    int64_t dim;
    if (!ParseInt64(&dim)) {
      return false;
    }
    tile_assignment_dimensions.push_back(dim);
  } while (EatIfPresent(TokKind::kComma));

  if (!ParseToken(TokKind::kRsquare,
                  "expected ']' to end sharding devices shape")) {
    return false;
  }
  if (lexer_.GetKind() == TokKind::kLeq) {
    lexer_.Lex();
    if (!ParseToken(TokKind::kLsquare,
                    "expected '[' to start sharding iota_reshape_dims")) {
      return false;
    }
    do {
      int64_t dim;
      if (!ParseInt64(&dim)) {
        return false;
      }
      iota_reshape_dims.push_back(dim);
    } while (EatIfPresent(TokKind::kComma));
    if (iota_reshape_dims.empty()) {
      return TokenError("expected non-empty iota_reshape_dims");
    }
    if (!ParseToken(TokKind::kRsquare,
                    "expected ']' to end sharding iota_reshape_dims")) {
      return false;
    }
    if (iota_reshape_dims.size() == 1) {
      iota_transpose_perm.push_back(0);
    } else {
      if (lexer_.GetKind() != TokKind::kIdent || lexer_.GetStrVal() != "T") {
        return TokenError(
            "expected 'T(' to start sharding devices "
            "iota_transpose_perm");
      }
      lexer_.Lex();
      if (!ParseToken(TokKind::kLparen,
                      "expected 'T(' to start sharding devices "
                      "iota_transpose_perm")) {
        return false;
      }
      do {
        int64_t dim;
        if (!ParseInt64(&dim)) {
          return false;
        }
        if (dim >= iota_reshape_dims.size()) {
          return TokenError(absl::StrFormat(
              "Out of range iota minor_to_major value %lld.", dim));
        }
        iota_transpose_perm.push_back(dim);
      } while (EatIfPresent(TokKind::kComma));
      if (!ParseToken(TokKind::kRparen,
                      "expected ')' to end sharding devices "
                      "iota_transpose_perm")) {
        return false;
      }
    }
  } else {
    if (!devices) {
      return TokenError(
          "Caller expected iota tile assignment when parsing, which should not "
          "have any manual device entries.");
    }
    do {
      int64_t device;
      if (!ParseInt64(&device)) {
        return false;
      }
      devices->push_back(device);
    } while (EatIfPresent(TokKind::kComma));
  }
  return true;
}

// ::= '{' 'replicated'? 'manual'? 'maximal'? 'unknown'? ('device=' int)? shape?
//         ('devices=' ('[' dims ']')* device_list)?
//         (('shard_like' | 'shard_as') int)* '}'
//         ('metadata=' metadata)*
//
// dims ::= int_list
// device_list ::= int_list? ('<=[' int_list ']{' int_list '}')?
// metadata ::= single_metadata |
//              ('{' [single_metadata (',' single_metadata)*] '}')
// last_tile_dims ::= sharding_type_list
bool HloParserImpl::ParseSingleSharding(std::optional<HloSharding>& sharding,
                                        bool lbrace_pre_lexed) {
  if (!lbrace_pre_lexed &&
      !ParseToken(TokKind::kLbrace,
                  "expected '{' to start sharding attribute")) {
    return false;
  }

  LocTy loc = lexer_.GetLoc();
  bool maximal = false;
  bool replicated = false;
  bool manual = false;
  bool unknown = false;
  bool last_tile_dim_replicate = false;
  bool last_tile_dims = false;
  bool shard_like = false;
  bool shard_as = false;
  int64_t shard_group_id;
  std::vector<int64_t> devices;
  std::vector<int64_t> tile_assignment_dimensions;
  std::vector<int64_t> iota_reshape_dims;
  std::vector<int> iota_transpose_perm;
  std::vector<OpSharding::Type> subgroup_types;
  std::vector<OpMetadata> metadata;
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
      case TokKind::kw_manual:
        manual = true;
        lexer_.Lex();
        break;
      case TokKind::kw_unknown:
        unknown = true;
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
          if (!ParseTileAssignment(tile_assignment_dimensions,
                                   iota_reshape_dims, iota_transpose_perm,
                                   &devices)) {
            return false;
          }
        } else if (lexer_.GetStrVal() == "metadata") {
          lexer_.Lex();
          if (!ParseSingleOrListMetadata(metadata)) {
            return false;
          }
        } else if (lexer_.GetStrVal() == "last_tile_dims") {
          last_tile_dims = true;
          lexer_.Lex();
          if (!ParseListShardingType(&subgroup_types)) {
            return false;
          }
        } else {
          return TokenError(
              "unknown attribute in sharding: expected device=, devices= "
              "metadata= or last_tile_dims= ");
        }
        break;
      }
      case TokKind::kw_last_tile_dim_replicate:
        last_tile_dim_replicate = true;
        lexer_.Lex();
        break;
      case TokKind::kw_shard_as: {
        shard_as = true;
        lexer_.Lex();
        if (!ParseInt64(&shard_group_id)) {
          return false;
        }
        break;
      }
      case TokKind::kw_shard_like: {
        shard_like = true;
        lexer_.Lex();
        if (!ParseInt64(&shard_group_id)) {
          return false;
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
    sharding = HloSharding::Replicate(metadata);
  } else if (maximal) {
    if (devices.size() != 1) {
      return Error(loc,
                   "maximal shardings should have exactly one device assigned");
    }
    sharding = HloSharding::AssignDevice(devices[0], metadata);
  } else if (manual) {
    if (!devices.empty()) {
      return Error(loc,
                   "manual shardings should not have any devices assigned");
    }
    sharding = HloSharding::Manual(metadata);
  } else if (unknown) {
    if (!devices.empty()) {
      return Error(loc,
                   "unknown shardings should not have any devices assigned");
    }
    sharding = HloSharding::Unknown(metadata);
  } else {
    if (tile_assignment_dimensions.empty()) {
      return Error(
          loc,
          "non-maximal shardings must have a tile assignment list including "
          "dimensions");
    }
    if (iota_transpose_perm.size() != iota_reshape_dims.size()) {
      return Error(loc,
                   absl::StrFormat(
                       "iota_transpose_perm should have the same rank as "
                       "iota_reshape_dims : expected %lld, saw %lld.",
                       iota_reshape_dims.size(), iota_transpose_perm.size()));
    }
    if (last_tile_dim_replicate) {
      CHECK(subgroup_types.empty());
      subgroup_types.push_back(OpSharding::REPLICATED);
    }
    if (!iota_reshape_dims.empty()) {
      CHECK(devices.empty());
      sharding =
          subgroup_types.empty()
              ? HloSharding::IotaTile(tile_assignment_dimensions,
                                      iota_reshape_dims, iota_transpose_perm,
                                      metadata)
              : HloSharding::Subgroup(
                    TileAssignment(tile_assignment_dimensions,
                                   iota_reshape_dims, iota_transpose_perm),
                    subgroup_types, metadata);
    } else {
      if (devices.size() <= 1) {
        return Error(
            loc,
            "non-maximal shardings must have more than one device assigned");
      }
      auto tiles = std::make_shared<Array<int64_t>>(tile_assignment_dimensions);
      absl::c_copy(devices, tiles->begin());
      sharding =
          subgroup_types.empty()
              ? HloSharding::Tile(TileAssignment(std::move(tiles)), metadata)
              : HloSharding::Subgroup(TileAssignment(std::move(tiles)),
                                      subgroup_types, metadata);
    }
  }

  if (shard_as || shard_like) {
    sharding = sharding->SetShardGroup(
        shard_as ? HloSharding::ShardAs(shard_group_id)
                 : HloSharding::ShardLike(shard_group_id));
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

// boolean_list ::=
//   ('true' | 'false') | ('{' ('true' | 'false')* (',' ('true' | 'false'))*
//   '}')
bool HloParserImpl::ParseBooleanListOrSingleBoolean(BoolList* boolean_list) {
  if (lexer_.GetKind() != TokKind::kLbrace &&
      lexer_.GetKind() != TokKind::kw_true &&
      lexer_.GetKind() != TokKind::kw_false) {
    TokenError("Expected list of booleans or true/false value");
    return false;
  }
  auto parse_boolean = [this, boolean_list]() {
    if (lexer_.GetKind() == TokKind::kw_true) {
      boolean_list->push_back(true);
      lexer_.Lex();
      return true;
    } else if (lexer_.GetKind() == TokKind::kw_false) {
      boolean_list->push_back(false);
      lexer_.Lex();
      return true;
    }
    return false;
  };
  if (parse_boolean()) {
    return true;
  }
  if (!ParseToken(TokKind::kLbrace,
                  "expected '{' to start boolean list attribute")) {
    return false;
  }
  if (lexer_.GetKind() != TokKind::kRbrace) {
    do {
      if (!parse_boolean()) {
        return false;
      }
    } while (EatIfPresent(TokKind::kComma));
  }

  return ParseToken(TokKind::kRbrace,
                    "expected '}' to end boolean list attribute");
}

// replica_groups ::='{' int64_tlist_elements '}'
// int64_tlist_elements
//   ::= /*empty*/
//   ::= int64_tlist (',' int64_tlist)*
// int64_tlist ::= '{' int64_elements '}'
// int64_elements
//   ::= /*empty*/
//   ::= int64_val (',' int64_val)*
bool HloParserImpl::ParseReplicaGroupsOnly(
    std::vector<ReplicaGroup>* replica_groups) {
  std::vector<std::vector<int64_t>> result;
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
  optional<HloSharding> entry_sharding;
  optional<HloSharding> exit_sharding;
  attrs["kind"] = {/*required=*/true, AttrTy::kString, &kind};
  attrs["entry"] = {/*required=*/true, AttrTy::kSharding, &entry_sharding};
  attrs["exit"] = {/*required=*/true, AttrTy::kSharding, &exit_sharding};
  if (!ParseSubAttributes(attrs)) {
    return false;
  }
  if (*kind == ShardingMetadata::KindName()) {
    auto entry_sharding_ptr =
        std::make_unique<HloSharding>(std::move(*entry_sharding));
    auto exit_sharding_ptr =
        std::make_unique<HloSharding>(std::move(*exit_sharding));
    domain->entry_metadata =
        std::make_unique<ShardingMetadata>(std::move(entry_sharding_ptr));
    domain->exit_metadata =
        std::make_unique<ShardingMetadata>(std::move(exit_sharding_ptr));
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

template <typename T>
std::string StringifyValue(T val) {
  if constexpr (is_complex_v<T>) {
    return StrFormat("(%f, %f)", val.real(), val.imag());
  } else {
    return StrCat(val);
  }
}

template <class T>
uint64_t GetNanPayload(T val) {
  if constexpr (std::is_same_v<T, double>) {
    auto rep = absl::bit_cast<uint64_t>(val);
    if (auto payload = rep & NanPayloadBitMask<double>()) {
      return payload;
    }
    return QuietNanWithoutPayload<double>();
  } else {
    static_assert(!std::numeric_limits<T>::has_quiet_NaN);
    static_assert(!std::numeric_limits<T>::has_signaling_NaN);
    return 0;
  }
}

template <typename LiteralNativeT, typename LiteralComponentT>
LiteralNativeT LiteralNativeFromRealImag(LiteralComponentT real,
                                         LiteralComponentT imag) {
  if constexpr (std::is_same_v<LiteralNativeT,
                               std::complex<LiteralComponentT>>) {
    return LiteralNativeT(real, imag);
  } else {
    return real;
  }
}

template <typename T>
struct ComponentType {
  using Type = T;
};

template <typename T>
struct ComponentType<std::complex<T>> {
  using Type = T;
};

template <typename T>
T GetReal(T value) {
  return value;
}

template <typename T>
T GetReal(std::complex<T> value) {
  return value.real();
}

template <typename T>
T GetImag(T value) {
  return 0;
}

template <typename T>
T GetImag(std::complex<T> value) {
  return value.imag();
}

// MaxFiniteValue is a type-traits helper used by
// HloParserImpl::CheckParsedValueIsInRange.
template <typename T>
struct MinMaxFiniteValue {
  static constexpr T max() { return std::numeric_limits<T>::max(); }
  static constexpr T min() { return std::numeric_limits<T>::lowest(); }
};

template <typename T>
bool IsFinite(T val) {
  if constexpr (std::numeric_limits<T>::has_infinity ||
                std::numeric_limits<T>::has_quiet_NaN ||
                std::numeric_limits<T>::has_signaling_NaN) {
    return Eigen::numext::isfinite(val);
  } else {
    return true;
  }
}

template <typename LiteralNativeT, typename ParsedElemT>
bool HloParserImpl::CheckParsedValueIsInRange(LocTy loc, ParsedElemT value) {
  if constexpr (std::is_floating_point_v<ParsedElemT>) {
    auto value_as_native_t = static_cast<LiteralNativeT>(value);
    auto value_double_converted = static_cast<ParsedElemT>(value_as_native_t);
    if (!IsFinite(value) || IsFinite(value_double_converted)) {
      value = value_double_converted;
    }
  }
  PrimitiveType literal_ty =
      primitive_util::NativeToPrimitiveType<LiteralNativeT>();
  if (!IsFinite(value)) {
    // Skip range checking for non-finite value.
  } else if constexpr (std::is_unsigned<LiteralNativeT>::value) {
    static_assert(std::is_same_v<ParsedElemT, int64_t> ||
                      std::is_same_v<ParsedElemT, bool>,
                  "Unimplemented checking for ParsedElemT");

    const uint64_t unsigned_value = value;
    const uint64_t upper_bound =
        static_cast<uint64_t>(std::numeric_limits<LiteralNativeT>::max());
    if (unsigned_value > upper_bound) {
      // Value is out of range for LiteralNativeT.
      return Error(loc, StrCat("value ", value,
                               " is out of range for literal's primitive type ",
                               PrimitiveType_Name(literal_ty), " namely [0, ",
                               upper_bound, "]."));
    }
  } else if (value > static_cast<ParsedElemT>(
                         MinMaxFiniteValue<LiteralNativeT>::max()) ||
             value < static_cast<ParsedElemT>(
                         MinMaxFiniteValue<LiteralNativeT>::min())) {
    // Value is out of range for LiteralNativeT.
    return Error(
        loc,
        StrCat(
            "value ", value, " is out of range for literal's primitive type ",
            PrimitiveType_Name(literal_ty), " namely [",
            static_cast<ParsedElemT>(MinMaxFiniteValue<LiteralNativeT>::min()),
            ", ",
            static_cast<ParsedElemT>(MinMaxFiniteValue<LiteralNativeT>::max()),
            "]."));
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
    if (!std::isfinite(v)) {
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

template <typename LiteralNativeT, typename ParsedElemT>
bool HloParserImpl::SetValueInLiteralHelper(LocTy loc, ParsedElemT value,
                                            int64_t index, Literal* literal) {
  if (!CheckParsedValueIsInRange<LiteralNativeT>(loc, value)) {
    return false;
  }

  // Check that the index is in range and assign into the literal
  if (index >= ShapeUtil::ElementsIn(literal->shape())) {
    return Error(loc, StrCat("tries to set value ", StringifyValue(value),
                             " to a literal in shape ",
                             ShapeUtil::HumanString(literal->shape()),
                             " at linear index ", index,
                             ", but the index is out of range"));
  }
  using ParsedElemComponentT = typename ComponentType<ParsedElemT>::Type;
  using LiteralNativeComponentT = typename ComponentType<LiteralNativeT>::Type;
  const auto handle_nan =
      [this, literal, index, loc](
          ParsedElemComponentT parsed_value_component,
          LiteralNativeComponentT* literal_value_component) {
        if (!std::isnan(static_cast<double>(parsed_value_component))) {
          return true;
        }
        auto nan_payload = GetNanPayload(parsed_value_component);
        if constexpr (NanPayloadBits<LiteralNativeComponentT>() > 0) {
          if (nan_payload == QuietNanWithoutPayload<double>()) {
            nan_payload = QuietNanWithoutPayload<LiteralNativeComponentT>();
          }
          const auto kLargestPayload =
              NanPayloadBitMask<LiteralNativeComponentT>();
          if (nan_payload > kLargestPayload) {
            return Error(
                loc, StrCat("tries to set NaN payload 0x",
                            absl::Hex(nan_payload), " to a literal in shape ",
                            ShapeUtil::HumanString(literal->shape()),
                            " at linear index ", index,
                            ", but the NaN payload is out of range (0x",
                            absl::Hex(kLargestPayload), ")"));
          }
          *literal_value_component =
              NanWithSignAndPayload<LiteralNativeComponentT>(
                  /*sign=*/std::signbit(
                      static_cast<double>(parsed_value_component)),
                  /*nan_payload=*/nan_payload);
        } else {
          if (nan_payload != QuietNanWithoutPayload<double>()) {
            return Error(
                loc, StrCat("tries to set NaN payload 0x",
                            absl::Hex(nan_payload), " to a literal in shape ",
                            ShapeUtil::HumanString(literal->shape()),
                            " at linear index ", index, ", but ",
                            primitive_util::LowercasePrimitiveTypeName(
                                literal->shape().element_type()),
                            " does not support payloads"));
          }
        }
        return true;
      };
  const ParsedElemComponentT parsed_real_value = GetReal(value);
  auto literal_real_value =
      static_cast<LiteralNativeComponentT>(parsed_real_value);
  if (std::is_floating_point_v<ParsedElemT> ||
      std::is_same_v<ParsedElemT, std::complex<double>>) {
    if (!handle_nan(parsed_real_value, &literal_real_value)) {
      return false;
    }
  }
  const ParsedElemComponentT parsed_imag_value = GetImag(value);
  auto literal_imag_value =
      static_cast<LiteralNativeComponentT>(parsed_imag_value);
  if constexpr (std::is_same_v<ParsedElemT, std::complex<double>>) {
    if (!handle_nan(parsed_real_value, &literal_imag_value)) {
      return false;
    }
  }
  literal->data<LiteralNativeT>().at(index) =
      LiteralNativeFromRealImag<LiteralNativeT>(literal_real_value,
                                                literal_imag_value);
  return true;
}

bool HloParserImpl::SetValueInLiteral(LocTy loc, int64_t value, int64_t index,
                                      Literal* literal) {
  const Shape& shape = literal->shape();
  return primitive_util::PrimitiveTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        if constexpr (primitive_type_constant == PRED) {
          return SetValueInLiteralHelper<bool>(loc, static_cast<bool>(value),
                                               index, literal);
        }
        if constexpr (primitive_util::IsIntegralType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          return SetValueInLiteralHelper<NativeT>(loc, value, index, literal);
        }
        LOG(FATAL) << "unknown integral primitive type "
                   << PrimitiveType_Name(shape.element_type());
      },
      shape.element_type());
}

bool HloParserImpl::SetValueInLiteral(LocTy loc, double value, int64_t index,
                                      Literal* literal) {
  const Shape& shape = literal->shape();
  return primitive_util::PrimitiveTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          return SetValueInLiteralHelper<NativeT>(loc, value, index, literal);
        }
        LOG(FATAL) << "unknown floating point primitive type "
                   << PrimitiveType_Name(shape.element_type());
      },
      shape.element_type());
}

bool HloParserImpl::SetValueInLiteral(LocTy loc, bool value, int64_t index,
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
                                      int64_t index, Literal* literal) {
  const Shape& shape = literal->shape();
  return primitive_util::PrimitiveTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        if constexpr (primitive_util::IsComplexType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          return SetValueInLiteralHelper<NativeT>(loc, value, index, literal);
        }
        LOG(FATAL) << PrimitiveType_Name(shape.element_type())
                   << " is not a complex type";
      },
      shape.element_type());
}

// Similar to ParseLiteral(Literal* literal, const Shape& shape), but parse the
// shape instead of accepting one as argument.
bool HloParserImpl::ParseLiteral(Literal* literal) {
  if (lexer_.GetKind() == TokKind::kLparen) {
    // Consume Lparen
    lexer_.Lex();
    std::vector<Literal> elements;
    while (lexer_.GetKind() != TokKind::kRparen) {
      Literal element;
      if (!ParseLiteral(&element)) {
        return TokenError("Fails when parsing tuple element");
      }
      elements.emplace_back(std::move(element));
      if (lexer_.GetKind() != TokKind::kRparen) {
        ParseToken(TokKind::kComma, "expects ',' to separate tuple elements");
      }
    }

    *literal = LiteralUtil::MakeTupleOwned(std::move(elements));
    // Consume Rparen
    return ParseToken(TokKind::kRparen, "expects ')' to close a tuple literal");
  }
  Shape literal_shape;
  if (!ParseShape(&literal_shape)) {
    return false;
  }
  return ParseLiteral(literal, literal_shape);
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
  // `rank` is an int64_t, that's an implicit narrowing conversion, which is
  // implementation-defined behavior.
  const int rank = static_cast<int>(shape.dimensions().size());

  // Create a literal with the given shape in default layout.
  *literal = LiteralUtil::CreateFromDimensions(shape.element_type(),
                                               shape.dimensions());
  int64_t nest_level = 0;
  int64_t linear_index = 0;
  // elems_seen_per_dim[i] is how many elements or sub-arrays we have seen for
  // the dimension i. For example, to parse f32[2,3] {{1, 2, 3}, {4, 5, 6}},
  // when we are parsing the 2nd '{' (right before '1'), we are seeing a
  // sub-array of the dimension 0, so elems_seen_per_dim[0]++. When we are at
  // the first '}' (right after '3'), it means the sub-array ends, and the
  // sub-array is supposed to contain exactly 3 elements, so check if
  // elems_seen_per_dim[1] is 3.
  std::vector<int64_t> elems_seen_per_dim(rank);
  auto get_index_str = [&elems_seen_per_dim](int dim) -> std::string {
    std::vector<int64_t> elems_seen_until_dim(elems_seen_per_dim.begin(),
                                              elems_seen_per_dim.begin() + dim);
    return StrCat("[",
                  StrJoin(elems_seen_until_dim, ",",
                          [](std::string* out, const int64_t num_elems) {
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
        if (nest_level == 0) {
          return TokenError("unexpected '}' token");
        }
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
              absl::StrFormat("unexpected '(' in literal. Parens are only "
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
        if (!options_.fill_shortform_constants_with_random_values()) {
          break;
        }
        // Fill data with deterministic (garbage) values. Use static to avoid
        // creating identical constants which could potentially got CSE'ed
        // away. This is a best-effort approach to make sure replaying a HLO
        // gives us same optimized HLO graph.
        static uint32_t data = 0;

        // According to the System V ABI not all 8 bit values are valid booleans
        // - only the values 0 and 1 are allowed. So to avoid undefined
        // behaviour we mask elements of type PRED accordingly. The mask assumes
        // that the C++ data type `bool` is represented as a single byte.
        static_assert(sizeof(bool) == 1);
        constexpr uint32_t kBooleanMask = 0x01010101;

        constexpr uint32_t kNoMask = 0xFFFFFFFF;
        const uint32_t mask =
            (shape.element_type() == PRED) ? kBooleanMask : kNoMask;

        uint32_t* raw_data = static_cast<uint32_t*>(literal->untyped_data());
        for (int64_t i = 0; i < literal->size_bytes() / 4; ++i) {
          raw_data[i] = data++ & mask;
        }
        uint8_t* raw_data_int8 = static_cast<uint8_t*>(literal->untyped_data());
        static uint8_t data_int8 = 0;
        for (int64_t i = 0; i < literal->size_bytes() % 4; ++i) {
          raw_data_int8[literal->size_bytes() / 4 + i] = data_int8++ & mask;
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
          int64_t value;
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

// operands ::= '(' operands1 ')'
// operands1
//   ::= /*empty*/
//   ::= operand (, operand)*
// operand ::= (shape)? name
//         ::= (shape)? opcode operands
bool HloParserImpl::ParseOperands(std::vector<HloInstruction*>* operands,
                                  HloComputation::Builder* builder) {
  CHECK(operands != nullptr);
  if (!ParseToken(TokKind::kLparen,
                  "expects '(' at the beginning of operands")) {
    return false;
  }
  if (lexer_.GetKind() == TokKind::kRparen) {
    // empty
  } else {
    do {
      // Try to parse the operand as a name with an optional shape.  If that
      // doesn't work, try again parsing it as a nested instruction.
      //
      // (Trying nested instructions second is important here: If you have a
      // giant HLO dump, it likely doesn't have any nested instructions, but
      // likely has tons of non-nested operands.  Generating an error is slow --
      // O(n) as of writing -- so we only want to hit the error branch in the
      // uncommon case.)
      HloLexer lexer_copy = lexer_;
      std::vector<std::string> saved_errors;
      std::swap(saved_errors, error_);
      bool is_normal_operand = [&] {
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
          // When parsing a single instruction (as opposed to a whole module),
          // an HLO may have one or more operands with a shape but no name:
          //
          //  foo = add(f32[10], f32[10])
          //
          // create_missing_instruction_ is always non-null when parsing a
          // single instruction, and is responsible for creating kParameter
          // instructions for these operands.
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

        // If this is a regular named operand, it must be followed by a comma or
        // a close-paren.  If not, it has to be a named instruction.  Don't
        // output an error here -- if it fails to parse as a named instruction
        // too, we'll just use that set of errors.
        auto next = lexer_.GetKind();
        if (next != TokKind::kComma && next != TokKind::kRparen) {
          return false;
        }

        operands->push_back(instruction->first);
        return true;
      }();

      if (is_normal_operand) {
        error_ = std::move(saved_errors);
        continue;
      }

      // If parsing as a normal operand failed, try parsing as a nested
      // instruction.
      std::vector<std::string> normal_operand_errors;
      std::swap(error_, normal_operand_errors);
      lexer_ = lexer_copy;

      // Nested instructions can't have attributes because it's ambiguous
      // whether the comma separates an instruction from its attribute, or
      // whether the comma separates two instructions.
      LocTy loc = lexer_.GetLoc();
      bool is_nested_instruction = ParseInstructionRhs(
          builder, /*name=*/"", loc, /*allow_attributes=*/false);
      if (is_nested_instruction) {
        operands->push_back(builder->last_added_instruction());
        error_ = std::move(saved_errors);
        continue;
      }

      // If neither parsing as a normal operand nor parsing as a nested
      // instruction worked, fail.  Return both sets of errors.
      std::vector<std::string> nested_instruction_errors;
      std::swap(error_, nested_instruction_errors);
      error_ = std::move(saved_errors);
      Error(loc,
            "cannot parse as an instruction name or as a nested instruction:");
      error_.insert(error_.end(),
                    std::make_move_iterator(normal_operand_errors.begin()),
                    std::make_move_iterator(normal_operand_errors.end()));
      error_.insert(error_.end(),
                    std::make_move_iterator(nested_instruction_errors.begin()),
                    std::make_move_iterator(nested_instruction_errors.end()));
    } while (EatIfPresent(TokKind::kComma));
  }
  return ParseToken(TokKind::kRparen, "expects ')' at the end of operands");
}

bool HloParserImpl::ParseOperands(std::vector<HloInstruction*>* operands,
                                  HloComputation::Builder* builder,
                                  const int expected_size) {
  CHECK(operands != nullptr);
  LocTy loc = lexer_.GetLoc();
  if (!ParseOperands(operands, builder)) {
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
    const absl::flat_hash_map<std::string, AttrConfig>& attrs,
    bool allow_attributes, const std::optional<Shape>& shape) {
  LocTy loc = lexer_.GetLoc();
  absl::flat_hash_set<std::string> seen_attrs;
  if (allow_attributes) {
    while (EatIfPresent(TokKind::kComma)) {
      if (!ParseAttributeHelper(attrs, &seen_attrs, shape)) {
        return false;
      }
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
    absl::flat_hash_set<std::string>* seen_attrs,
    const std::optional<Shape>& shape) {
  LocTy loc = lexer_.GetLoc();
  std::string name;
  if (!ParseAttributeName(&name)) {
    return Error(loc, "error parsing attributes");
  }
  VLOG(kDebugLevel) << "Parsing attribute " << name;
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
    return Error(
        loc, StrFormat("unexpected attribute \"%s\". %s", name, allowed_attrs));
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
      case AttrTy::kBracedBoolListOrBool: {
        if (!ParseBooleanListOrSingleBoolean(
                static_cast<BoolList*>(attr_out_ptr))) {
          return false;
        }
        return true;
      }
      case AttrTy::kInt64: {
        int64_t result;
        if (!ParseInt64(&result)) {
          return false;
        }
        static_cast<optional<int64_t>*>(attr_out_ptr)->emplace(result);
        return true;
      }
      case AttrTy::kInt32: {
        int64_t result;
        if (!ParseInt64(&result)) {
          return false;
        }
        if (result != static_cast<int32_t>(result)) {
          return Error(attr_loc, "value out of range for int32_t");
        }
        static_cast<optional<int32_t>*>(attr_out_ptr)
            ->emplace(static_cast<int32_t>(result));
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
      case AttrTy::kPaddingType: {
        PaddingType result;
        if (!ParsePaddingType(&result)) {
          return false;
        }
        static_cast<optional<PaddingType>*>(attr_out_ptr)->emplace(result);
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
      case AttrTy::kComparisonType: {
        Comparison::Type result;
        if (!ParseComparisonType(&result)) {
          return false;
        }
        static_cast<optional<Comparison::Type>*>(attr_out_ptr)->emplace(result);
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
        std::optional<HloSharding> sharding;
        if (!ParseSharding(sharding)) {
          return false;
        }
        static_cast<optional<HloSharding>*>(attr_out_ptr)
            ->emplace(std::move(*sharding));
        return true;
      }
      case AttrTy::kCollectiveDeviceList: {
        CollectiveDeviceList device_list;
        if (!ParseCollectiveDeviceList(&device_list)) {
          return false;
        }
        *(static_cast<CollectiveDeviceList*>(attr_out_ptr)) = device_list;
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
      case AttrTy::kStatisticsViz: {
        StatisticsViz statistics_viz;
        if (!ParseStatisticsViz(&statistics_viz)) {
          return false;
        }
        static_cast<optional<StatisticsViz>*>(attr_out_ptr)
            ->emplace(statistics_viz);
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
        std::vector<int64_t> result;
        if (!ParseInt64List(TokKind::kLbrace, TokKind::kRbrace, TokKind::kComma,
                            &result)) {
          return false;
        }
        static_cast<optional<std::vector<int64_t>>*>(attr_out_ptr)
            ->emplace(result);
        return true;
      }
      case AttrTy::kBracedInt64ListList: {
        std::vector<std::vector<int64_t>> result;
        if (!ParseInt64ListList(TokKind::kLbrace, TokKind::kRbrace,
                                TokKind::kComma, &result)) {
          return false;
        }
        static_cast<optional<std::vector<std::vector<int64_t>>>*>(attr_out_ptr)
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
        static_cast<optional<std::string>*>(attr_out_ptr)
            ->emplace(std::move(result));
        return true;
      }
      case AttrTy::kStringOrJsonDict: {
        std::string result;
        if (lexer_.GetKind() == TokKind::kString) {
          if (!ParseString(&result)) {
            return false;
          }
        } else if (lexer_.GetKind() == TokKind::kLbrace) {
          if (!ParseJsonDict(&result)) {
            return false;
          }
        } else {
          return false;
        }
        static_cast<optional<std::string>*>(attr_out_ptr)
            ->emplace(std::move(result));
        return true;
      }
      case AttrTy::kOriginalValue: {
        // By the time this attribute is added, the instruction shape should
        // have been inferred.
        if (!shape) {
          return TokenError("expects instruction shape");
        }
        return ParseOriginalValue(
            static_cast<optional<std::shared_ptr<OriginalValue>>*>(
                attr_out_ptr),
            *shape);
      }
      case AttrTy::kMetadata: {
        OpMetadata result;
        if (!ParseMetadata(result)) {
          return false;
        }
        static_cast<optional<OpMetadata>*>(attr_out_ptr)
            ->emplace(std::move(result));
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
      case AttrTy::kShape: {
        Shape result;
        if (!ParseShape(&result)) {
          return false;
        }
        static_cast<optional<Shape>*>(attr_out_ptr)->emplace(result);
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
      case AttrTy::kPrecisionAlgorithm: {
        PrecisionConfig::Algorithm result;
        if (!ParseAlgorithm(&result)) {
          return false;
        }
        static_cast<optional<PrecisionConfig::Algorithm>*>(attr_out_ptr)
            ->emplace(result);
        return true;
      }
      case AttrTy::kResultAccuracyType: {
        ResultAccuracy::Mode result;
        if (!ParseResultAccuracyType(&result)) {
          return false;
        }
        static_cast<optional<ResultAccuracy::Mode>*>(attr_out_ptr)
            ->emplace(result);
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
      case AttrTy::kBufferDonor: {
        BufferDonor buffer_donor;
        if (!ParseBufferDonor(&buffer_donor)) {
          return false;
        }
        static_cast<optional<BufferDonor>*>(attr_out_ptr)
            ->emplace(buffer_donor);
        return true;
      }
      case AttrTy::kComputationLayout: {
        ComputationLayout computation_layout(ShapeLayout(Shape{}));
        if (!ParseComputationLayout(&computation_layout)) {
          return false;
        }
        static_cast<optional<ComputationLayout>*>(attr_out_ptr)
            ->emplace(computation_layout);
        return true;
      }
      case AttrTy::kInstructionAliasing: {
        std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
            aliasing_output_operand_pairs;
        if (!ParseInstructionOutputOperandAliasing(
                &aliasing_output_operand_pairs)) {
          return false;
        }
        static_cast<optional<std::vector<
            std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>>*>(
            attr_out_ptr)
            ->emplace(std::move(aliasing_output_operand_pairs));
        return true;
      }
      case AttrTy::kLiteral: {
        Literal result;
        if (!ParseLiteral(&result)) {
          return false;
        }
        static_cast<optional<Literal>*>(attr_out_ptr)
            ->emplace(std::move(result));
        return true;
      }
      case AttrTy::kCustomCallSchedule: {
        CustomCallSchedule result;
        if (!ParseCustomCallSchedule(&result)) {
          return false;
        }
        static_cast<optional<CustomCallSchedule>*>(attr_out_ptr)
            ->emplace(result);
        return true;
      }
      case AttrTy::kCustomCallApiVersion: {
        CustomCallApiVersion result;
        if (!ParseCustomCallApiVersion(&result)) {
          return false;
        }
        static_cast<optional<CustomCallApiVersion>*>(attr_out_ptr)
            ->emplace(result);
        return true;
      }
      case AttrTy::kSparsityDescriptor: {
        std::vector<SparsityDescriptor> result;
        if (!ParseSparsityDescriptor(&result)) {
          return false;
        }
        *static_cast<std::vector<SparsityDescriptor>*>(attr_out_ptr) =
            std::move(result);
        return true;
      }
      case AttrTy::kResultAccuracy: {
        ResultAccuracy result;
        if (!ParseResultAccuracy(&result)) {
          return false;
        }
        static_cast<optional<ResultAccuracy>*>(attr_out_ptr)->emplace(result);
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
    tsl::protobuf::Message* message) {
  const tsl::protobuf::Descriptor* descriptor = message->GetDescriptor();
  const tsl::protobuf::Reflection* reflection = message->GetReflection();

  for (const auto& p : attrs) {
    const std::string& name = p.first;
    if (non_proto_attrs.find(name) != non_proto_attrs.end()) {
      continue;
    }
    const tsl::protobuf::FieldDescriptor* fd =
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
          StrFormat("unexpected attribute \"%s\". %s", name, allowed_attrs));
    }

    CHECK(!fd->is_repeated());  // Repeated fields not implemented.
    bool success = [&] {
      switch (fd->type()) {
        case tsl::protobuf::FieldDescriptor::TYPE_BOOL: {
          auto attr_value = static_cast<optional<bool>*>(p.second.result);
          if (attr_value->has_value()) {
            reflection->SetBool(message, fd, **attr_value);
          }
          return true;
        }
        case tsl::protobuf::FieldDescriptor::TYPE_ENUM: {
          auto attr_value =
              static_cast<optional<std::string>*>(p.second.result);
          if (attr_value->has_value()) {
            const tsl::protobuf::EnumValueDescriptor* evd =
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
    tsl::protobuf::Message* message) {
  const tsl::protobuf::Descriptor* descriptor = message->GetDescriptor();
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
    const tsl::protobuf::FieldDescriptor* fd = descriptor->field(field_idx);
    absl::string_view field_name = fd->name();
    switch (fd->type()) {
      case tsl::protobuf::FieldDescriptor::TYPE_BOOL: {
        bool_params.emplace_back(std::nullopt);
        attrs[field_name] = {/*is_required*/ false, AttrTy::kBool,
                             &bool_params.back()};
        break;
      }
      case tsl::protobuf::FieldDescriptor::TYPE_ENUM: {
        string_params.emplace_back(std::nullopt);
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
      tsl::gtl::FindOrNull(computation_pool_, name);
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

  std::vector<int64_t> size;
  std::vector<int64_t> stride;
  std::vector<std::vector<int64_t>> pad;
  std::vector<int64_t> lhs_dilate;
  std::vector<int64_t> rhs_dilate;
  std::vector<int64_t> rhs_reversal;
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
// The string looks like "dim_labels=0bf_0io->0bf".
//
// '?' dims don't appear in ConvolutionDimensionNumbers.  There can be more than
// one '?' dim.
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

  auto is_unique = [](absl::string_view str) -> bool {
    absl::flat_hash_set<char> chars;
    for (char c : str) {
      // '?' dims are skipped.
      if (c == '?') {
        continue;
      }
      if (!chars.insert(c).second) {
        return false;
      }
    }
    return true;
  };

  // lhs
  {
    if (!is_unique(lhs)) {
      return TokenError(
          StrCat("expects unique lhs dimension numbers, but sees ", lhs));
    }
    // Count number of spatial dimensions.
    for (char c : lhs) {
      if (c != 'b' && c != 'f' && c != '?') {
        dnums->add_input_spatial_dimensions(-1);
      }
    }
    for (int i = 0; i < lhs.size(); i++) {
      char c = lhs[i];
      if (c == '?') {
        continue;
      } else if (c == 'b') {
        dnums->set_input_batch_dimension(i);
      } else if (c == 'f') {
        dnums->set_input_feature_dimension(i);
      } else if (c < '0' + lhs.size() && c >= '0') {
        dnums->set_input_spatial_dimensions(c - '0', i);
      } else {
        return TokenError(StrFormat(
            "expects [0-%dbf?] in lhs dimension numbers", lhs.size() - 1));
      }
    }
  }
  // rhs
  {
    if (!is_unique(rhs)) {
      return TokenError(
          StrCat("expects unique rhs dimension numbers, but sees ", rhs));
    }
    // Count number of spatial dimensions.
    for (char c : rhs) {
      if (c != 'i' && c != 'o' && c != '?') {
        dnums->add_kernel_spatial_dimensions(-1);
      }
    }
    for (int i = 0; i < rhs.size(); i++) {
      char c = rhs[i];
      if (c == '?') {
        continue;
      } else if (c == 'i') {
        dnums->set_kernel_input_feature_dimension(i);
      } else if (c == 'o') {
        dnums->set_kernel_output_feature_dimension(i);
      } else if (c < '0' + rhs.size() && c >= '0') {
        dnums->set_kernel_spatial_dimensions(c - '0', i);
      } else {
        return TokenError(StrFormat(
            "expects [0-%dio?] in rhs dimension numbers", rhs.size() - 1));
      }
    }
  }
  // output
  {
    if (!is_unique(out)) {
      return TokenError(
          StrCat("expects unique output dimension numbers, but sees ", out));
    }
    // Count number of spatial dimensions.
    for (char c : out) {
      if (c != 'b' && c != 'f' && c != '?') {
        dnums->add_output_spatial_dimensions(-1);
      }
    }
    for (int i = 0; i < out.size(); i++) {
      char c = out[i];
      if (c == '?') {
        continue;
      } else if (c == 'b') {
        dnums->set_output_batch_dimension(i);
      } else if (c == 'f') {
        dnums->set_output_feature_dimension(i);
      } else if (c < '0' + out.size() && c >= '0') {
        dnums->set_output_spatial_dimensions(c - '0', i);
      } else {
        return TokenError(StrFormat(
            "expects [0-%dbf?] in output dimension numbers", out.size() - 1));
      }
    }
  }

  // lhs, rhs, and output should have the same number of spatial dimensions.
  if (dnums->input_spatial_dimensions_size() !=
          dnums->output_spatial_dimensions_size() ||
      dnums->input_spatial_dimensions_size() !=
          dnums->kernel_spatial_dimensions_size()) {
    return TokenError(
        StrFormat("input, kernel, and output must have same number of spatial "
                  "dimensions, but got %d, %d, %d, respectively.",
                  dnums->input_spatial_dimensions_size(),
                  dnums->kernel_spatial_dimensions_size(),
                  dnums->output_spatial_dimensions_size()));
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
  std::vector<std::vector<int64_t>> ranges;
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
    VLOG(kDebugLevel) << "parsed computation " << computation->name();
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

// int64_tlist ::= start int64_elements end
// int64_elements
//   ::= /*empty*/
//   ::= int64_val (delim int64_val)*
bool HloParserImpl::ParseInt64List(const TokKind start, const TokKind end,
                                   const TokKind delim,
                                   std::vector<int64_t>* result) {
  auto parse_and_add_item = [&]() {
    int64_t i;
    if (!ParseInt64(&i)) {
      return false;
    }
    result->push_back(i);
    return true;
  };
  return ParseList(start, end, delim, parse_and_add_item);
}

// int64_tlistlist ::= start int64_tlist_elements end
// int64_tlist_elements
//   ::= /*empty*/
//   ::= int64_tlist (delim int64_tlist)*
// int64_tlist ::= start int64_elements end
// int64_elements
//   ::= /*empty*/
//   ::= int64_val (delim int64_val)*
bool HloParserImpl::ParseInt64ListList(
    const TokKind start, const TokKind end, const TokKind delim,
    std::vector<std::vector<int64_t>>* result) {
  auto parse_and_add_item = [&]() {
    std::vector<int64_t> item;
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
                              absl::FunctionRef<bool()> parse_and_add_item) {
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
//   ::= '?'
//   ::= <=? int64_t (',' param)*
// param ::= name shape
bool HloParserImpl::ParseDimensionSizes(std::vector<int64_t>* dimension_sizes,
                                        std::vector<bool>* dynamic_dimensions) {
  auto parse_and_add_item = [&]() {
    int64_t i;
    bool is_dynamic = false;
    if (lexer_.GetKind() == TokKind::kQuestionMark) {
      i = Shape::kUnboundedSize;
      is_dynamic = true;
      lexer_.Lex();
    } else {
      if (lexer_.GetKind() == TokKind::kLeq) {
        is_dynamic = true;
        lexer_.Lex();
      }
      if (!ParseInt64(&i)) {
        return false;
      }
    }
    dimension_sizes->push_back(i);
    dynamic_dimensions->push_back(is_dynamic);
    return true;
  };
  return ParseList(TokKind::kLsquare, TokKind::kRsquare, TokKind::kComma,
                   parse_and_add_item);
}

// dim_level_types
//   ::=  /* empty */
//   ::= 'D' '(' dim_level_type_list ')'
// dim_level_type_list
//   ::= /* empty */
//   ..= dim_level_type (',' dim_level_type)*
// dim_level_type
//   ::= 'D'
//   ::= 'C'
//   ::= 'S'
bool HloParserImpl::ParseDimLevelTypes(
    absl::InlinedVector<DimLevelType, InlineRank()>* dim_level_types,
    absl::InlinedVector<bool, InlineRank()>* dim_unique,
    absl::InlinedVector<bool, InlineRank()>* dim_ordered) {
  auto parse_and_add_item = [&]() {
    if (lexer_.GetKind() == TokKind::kIdent) {
      bool dim_level_type_valid = false;
      DimLevelType dim_level_type;
      if (lexer_.GetStrVal() == "D") {
        lexer_.Lex();
        dim_level_type = DIM_DENSE;
        dim_level_type_valid = true;
      } else if (lexer_.GetStrVal() == "C") {
        lexer_.Lex();
        dim_level_type = DIM_COMPRESSED;
        dim_level_type_valid = true;
      } else if (lexer_.GetStrVal() == "S") {
        lexer_.Lex();
        dim_level_type = DIM_SINGLETON;
        dim_level_type_valid = true;
      } else if (lexer_.GetStrVal() == "H") {
        lexer_.Lex();
        dim_level_type = DIM_LOOSE_COMPRESSED;
        dim_level_type_valid = true;
      }
      if (dim_level_type_valid) {
        bool new_dim_unique = true;
        if (lexer_.GetKind() == TokKind::kPlus) {
          new_dim_unique = false;
          lexer_.Lex();
        }
        bool new_dim_ordered = true;
        if (lexer_.GetKind() == TokKind::kTilde) {
          new_dim_ordered = false;
          lexer_.Lex();
        }
        if (!LayoutUtil::ValidateDimLevel(dim_level_type, new_dim_unique,
                                          new_dim_ordered)) {
          return Error(
              lexer_.GetLoc(),
              "invalid DimLevelType/unique/ordered combination in shape");
        }
        dim_level_types->push_back(dim_level_type);
        dim_unique->push_back(new_dim_unique);
        dim_ordered->push_back(new_dim_ordered);
        return true;
      }
    }
    return Error(lexer_.GetLoc(),
                 "expected a DimLevelType abbreviation (D, C, or S)");
  };
  return ParseList(TokKind::kLparen, TokKind::kRparen, TokKind::kComma,
                   parse_and_add_item);
}

// tiles
//   ::= /*empty*/
//   ::= 'T' ('(' dim_list ')')+
// dim_list
//   ::= /*empty*/
//   ::= (int64_t | '*') (',' (int64_t | '*'))*
bool HloParserImpl::ParseTiles(std::vector<Tile>* tiles) {
  auto parse_and_add_tile_dimension = [&]() {
    int64_t i;
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

// physical_shape
//   ::= /*empty*/
//   ::= 'P' '(' shape ')'
bool HloParserImpl::ParsePhysicalShape(Shape* physical_shape) {
  if (!ParseToken(TokKind::kLparen,
                  StrCat("expects physical shape to start with ",
                         TokKindToString(TokKind::kLparen)))) {
    return false;
  }
  ParseShape(physical_shape);
  if (!ParseToken(TokKind::kRparen,
                  StrCat("expects physical shape to end with ",
                         TokKindToString(TokKind::kRparen)))) {
    return false;
  }
  return true;
}

bool HloParserImpl::ParsePrimitiveType(PrimitiveType* result) {
  if (lexer_.GetKind() != TokKind::kPrimitiveType) {
    return TokenError(absl::StrCat("expected primitive type, saw ",
                                   TokKindToString(lexer_.GetKind())));
  }
  *result = lexer_.GetPrimitiveTypeVal();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseUnsignedIntegerType(PrimitiveType* primitive_type) {
  if (!ParsePrimitiveType(primitive_type)) {
    return false;
  }
  if (!primitive_util::IsUnsignedIntegralType(*primitive_type)) {
    return TokenError("expecting an unsigned integer type");
  }
  return true;
}

// int_attribute
//   ::= /*empty*/
//   ::= attr_token '(' attr_value ')'
// attr_token
//   ::= 'E' | 'S'
// attr_value
//   ::= int64_t
bool HloParserImpl::ParseLayoutIntAttribute(
    int64_t* attr_value, absl::string_view attr_description) {
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

// split_configs
//   ::= /*empty*/
//   ::= 'SC' ('(' int64_t ':' int64_list ')')+
bool HloParserImpl::ParseSplitConfigs(std::vector<SplitConfig>& split_configs) {
  auto parse_and_add_split_index = [&]() {
    int64_t i;
    if (ParseInt64(&i)) {
      split_configs.back().add_split_indices(i);
      return true;
    }
    return false;
  };

  do {
    if (!ParseToken(TokKind::kLparen,
                    StrCat("expects split configs to start with ",
                           TokKindToString(TokKind::kLparen)))) {
      return false;
    }
    int64_t dimension;
    if (!ParseInt64(&dimension)) {
      return false;
    }
    split_configs.push_back(SplitConfig(dimension, {}));
    if (!ParseList(TokKind::kColon, TokKind::kRparen, TokKind::kComma,
                   parse_and_add_split_index)) {
      return false;
    }
  } while (lexer_.GetKind() == TokKind::kLparen);
  return true;
}

// layout
//   ::= '{' int64_list
//       (':' dim_level_types
//            tiles
//            tail_padding_alignment_in_elements
//            element_size_in_bits
//            memory_space
//            split_configs
//            physical_shape
//            dynamic_shape_metadata_prefix_bytes)?
//       '}'
// element_size_in_bits
//   ::= /*empty*/
//   ::= 'E' '(' int64_t ')'
// memory_space
//   ::= /*empty*/
//   ::= 'S' '(' int64_t ')'
bool HloParserImpl::ParseLayout(Layout* layout) {
  absl::InlinedVector<int64_t, InlineRank()> minor_to_major;
  DimLevelTypeVector dim_level_types;
  absl::InlinedVector<bool, InlineRank()> dim_unique;
  absl::InlinedVector<bool, InlineRank()> dim_ordered;
  std::vector<Tile> tiles;
  PrimitiveType index_primitive_type = PRIMITIVE_TYPE_INVALID;
  PrimitiveType pointer_primitive_type = PRIMITIVE_TYPE_INVALID;
  int64_t element_size_in_bits = 0;
  int64_t memory_space = 0;
  std::vector<SplitConfig> split_configs;
  std::optional<Shape> physical_shape;
  int64_t dynamic_shape_metadata_prefix_bytes = 0;
  int64_t tail_padding_alignment_in_elements = 1;

  auto parse_and_add_item = [&]() {
    int64_t i;
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

      if (lexer_.GetKind() == TokKind::kIdent && lexer_.GetStrVal() == "D") {
        lexer_.Lex();
        ParseDimLevelTypes(&dim_level_types, &dim_unique, &dim_ordered);
      }

      if (lexer_.GetKind() == TokKind::kIdent && lexer_.GetStrVal() == "T") {
        lexer_.Lex();
        ParseTiles(&tiles);
      }

      if (lexer_.GetKind() == TokKind::kIdent && lexer_.GetStrVal() == "L") {
        lexer_.Lex();
        ParseLayoutIntAttribute(&tail_padding_alignment_in_elements,
                                "multiple padded to in elements");
      }

      if (lexer_.GetKind() == TokKind::kOctothorp) {
        lexer_.Lex();
        ParseToken(
            TokKind::kLparen,
            StrCat("expects ", TokKindToString(TokKind::kOctothorp),
                   " to be followed by ", TokKindToString(TokKind::kLparen)));
        ParseUnsignedIntegerType(&index_primitive_type);
        ParseToken(TokKind::kRparen,
                   StrCat("expects index primitive type to be followed by ",
                          TokKindToString(TokKind::kRparen)));
      }

      if (lexer_.GetKind() == TokKind::kAsterisk) {
        lexer_.Lex();
        ParseToken(
            TokKind::kLparen,
            StrCat("expects ", TokKindToString(TokKind::kAsterisk),
                   " to be followed by ", TokKindToString(TokKind::kLparen)));
        ParseUnsignedIntegerType(&pointer_primitive_type);
        ParseToken(TokKind::kRparen,
                   StrCat("expects pointer primitive type to be followed by ",
                          TokKindToString(TokKind::kRparen)));
      }

      if (lexer_.GetKind() == TokKind::kIdent && lexer_.GetStrVal() == "E") {
        lexer_.Lex();
        ParseLayoutIntAttribute(&element_size_in_bits, "element size in bits");
      }

      if (lexer_.GetKind() == TokKind::kIdent && lexer_.GetStrVal() == "S") {
        lexer_.Lex();
        ParseLayoutIntAttribute(&memory_space, "memory space");
      }

      if (lexer_.GetKind() == TokKind::kIdent && lexer_.GetStrVal() == "SC") {
        lexer_.Lex();
        ParseSplitConfigs(split_configs);
      }

      if (lexer_.GetKind() == TokKind::kIdent && lexer_.GetStrVal() == "P") {
        lexer_.Lex();
        physical_shape.emplace();
        ParsePhysicalShape(&*physical_shape);
      }

      if (lexer_.GetKind() == TokKind::kIdent && lexer_.GetStrVal() == "M") {
        lexer_.Lex();
        ParseLayoutIntAttribute(&dynamic_shape_metadata_prefix_bytes,
                                "dynamic shape metadata prefix bytes");
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
  *layout = LayoutUtil::MakeLayout(
      minor_to_major, dim_level_types, dim_unique, dim_ordered, vec_tiles,
      tail_padding_alignment_in_elements, index_primitive_type,
      pointer_primitive_type, element_size_in_bits, memory_space, split_configs,
      std::move(physical_shape), dynamic_shape_metadata_prefix_bytes);
  return true;
}

// shape ::= shape_val_
// shape ::= '(' tuple_elements ')'
// tuple_elements
//   ::= /*empty*/
//   ::= shape (',' shape)*
bool HloParserImpl::ParseShape(Shape* result,
                               bool allow_fallback_to_default_layout) {
  if (EatIfPresent(TokKind::kLparen)) {  // Tuple
    std::vector<Shape> shapes;
    if (lexer_.GetKind() == TokKind::kRparen) {
      /*empty*/
    } else {
      // shape (',' shape)*
      do {
        shapes.emplace_back();
        if (!ParseShape(&shapes.back(), allow_fallback_to_default_layout)) {
          return false;
        }
      } while (EatIfPresent(TokKind::kComma));
    }
    *result = ShapeUtil::MakeTupleShape(shapes);
    return ParseToken(TokKind::kRparen, "expects ')' at the end of tuple.");
  }

  PrimitiveType primitive_type;
  if (!ParsePrimitiveType(&primitive_type)) {
    return false;
  }

  // Each element contains a dimension size and a bool indicating whether this
  // is a dynamic dimension.
  std::vector<int64_t> dimension_sizes;
  std::vector<bool> dynamic_dimensions;
  if (!ParseDimensionSizes(&dimension_sizes, &dynamic_dimensions)) {
    return false;
  }
  result->set_element_type(primitive_type);
  for (int i = 0; i < dimension_sizes.size(); ++i) {
    if (!Shape::IsValidDimensionSize(dimension_sizes[i],
                                     dynamic_dimensions[i])) {
      return false;
    }
    result->add_dimensions(dimension_sizes[i], dynamic_dimensions[i]);
  }
  if ((allow_fallback_to_default_layout && options_.fill_missing_layouts()) ||
      ShapeUtil::IsScalar(*result)) {
    LayoutUtil::SetToDefaultLayout(result);
  }
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
    if (layout.dim_level_types_size() != 0 &&
        layout.dim_level_types_size() != result->dimensions().size()) {
      return Error(
          lexer_.GetLoc(),
          StrFormat("Dimensions size is %ld, but dim level types size is %ld.",
                    result->dimensions().size(),
                    layout.dim_level_types_size()));
    }
    if (layout.minor_to_major_size() != result->dimensions().size()) {
      return Error(
          lexer_.GetLoc(),
          StrFormat("Dimensions size is %ld, but minor to major size is %ld.",
                    result->dimensions().size(), layout.minor_to_major_size()));
    }
    if (LayoutUtil::IsSparse(layout) && layout.tiles_size() > 0) {
      return Error(lexer_.GetLoc(),
                   StrFormat("Layout has tiles, but is for a sparse array: %s",
                             layout.ToString()));
    }
    if (!LayoutUtil::IsSparse(layout) && layout.has_physical_shape()) {
      return Error(
          lexer_.GetLoc(),
          StrFormat(
              "Layout has physical shape, but is not for a sparse array: %s",
              layout.ToString()));
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
  VLOG(kDebugLevel) << "ParseName";
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
  VLOG(kDebugLevel) << "ParseString";
  if (lexer_.GetKind() != TokKind::kString) {
    return TokenError("expects string");
  }
  *result = lexer_.GetStrVal();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseJsonDict(std::string* result) {
  VLOG(kDebugLevel) << "ParseJsonDict";
  if (lexer_.LexJsonDict() != TokKind::kString) {
    return TokenError("expects JSON dict");
  }
  *result = lexer_.GetStrVal();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseDxD(const std::string& name,
                             std::vector<int64_t>* result) {
  LocTy loc = lexer_.GetLoc();
  if (!result->empty()) {
    return Error(loc, StrFormat("sub-attribute '%s=' already exists", name));
  }
  // 1D
  if (lexer_.GetKind() == TokKind::kInt) {
    int64_t number;
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

bool HloParserImpl::ParseWindowPad(std::vector<std::vector<int64_t>>* pad) {
  LocTy loc = lexer_.GetLoc();
  if (!pad->empty()) {
    return Error(loc, "sub-attribute 'pad=' already exists");
  }
  if (lexer_.GetKind() != TokKind::kPad) {
    return TokenError("expects window pad pattern, e.g., '0_0x3_3'");
  }
  std::string str = lexer_.GetStrVal();
  for (const auto& padding_dim_str : absl::StrSplit(str, 'x')) {
    std::vector<int64_t> low_high;
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
    std::vector<int64_t> padding_dim;
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

// original_value ::= original_value | '{' [shape_index] ',' original_array '}'
// [',']
bool HloParserImpl::ParseOriginalValue(
    optional<std::shared_ptr<OriginalValue>>* original_value,
    const Shape& shape) {
  VLOG(kDebugLevel) << "ParseOriginalValue";

  if (!ParseToken(TokKind::kLbrace, "Expects '{'")) {
    return false;
  }

  *original_value = std::make_shared<OriginalValue>(shape);

  ShapeIndex leaf_shape_index;
  while (lexer_.GetKind() != TokKind::kRbrace) {
    if (lexer_.GetKind() == TokKind::kLparen) {
      lexer_.Lex();
      leaf_shape_index.push_back(0);
    } else if (lexer_.GetKind() == TokKind::kRparen) {
      lexer_.Lex();
      leaf_shape_index.pop_back();
    } else if (lexer_.GetKind() == TokKind::kComma) {
      lexer_.Lex();
      ++leaf_shape_index.back();
    } else if (lexer_.GetKind() == TokKind::kLbrace) {
      lexer_.Lex();
      if (lexer_.GetKind() != TokKind::kRbrace) {
        std::string instruction_name;
        ShapeIndex shape_index;
        if (!ParseString(&instruction_name)) {
          return false;
        }
        if (lexer_.GetKind() != TokKind::kRbrace) {
          if (!ParseShapeIndex(&shape_index)) {
            return false;
          }
        }
        *(**original_value)->mutable_element(leaf_shape_index) = {
            instruction_name, shape_index};
      } else {
        // The original_value is not expected to have any leaf without values.
        // However we should not fail the execution here. This should
        // be done in HloVerifier instead.
        LOG(WARNING) << "Found an empty leaf node in an original value";
      }
      if (!ParseToken(TokKind::kRbrace,
                      "Expects '} at end of each OriginalArray'")) {
        return false;
      }
    } else {
      return false;
    }
  }

  lexer_.Lex();
  return true;
}

// '{' metadata_string '}'
bool HloParserImpl::ParseMetadata(OpMetadata& metadata) {
  absl::flat_hash_map<std::string, AttrConfig> attrs;
  optional<std::string> op_type;
  optional<std::string> op_name;
  optional<std::string> source_file;
  optional<int32_t> source_line;
  optional<std::vector<int64_t>> profile_type;
  optional<std::string> deduplicated_name;
  optional<std::string> scheduling_name;
  attrs["op_type"] = {/*required=*/false, AttrTy::kString, &op_type};
  attrs["op_name"] = {/*required=*/false, AttrTy::kString, &op_name};
  attrs["source_file"] = {/*required=*/false, AttrTy::kString, &source_file};
  attrs["source_line"] = {/*required=*/false, AttrTy::kInt32, &source_line};
  attrs["profile_type"] = {/*required=*/false, AttrTy::kBracedInt64List,
                           &profile_type};
  attrs["deduplicated_name"] = {/*required=*/false, AttrTy::kString,
                                &deduplicated_name};
  attrs["scheduling_name"] = {/*required=*/false, AttrTy::kString,
                              &scheduling_name};
  if (!ParseSubAttributes(attrs)) {
    return false;
  }
  if (op_type) {
    metadata.set_op_type(*op_type);
  }
  if (op_name) {
    metadata.set_op_name(*op_name);
  }
  if (source_file) {
    metadata.set_source_file(*source_file);
  }
  if (source_line) {
    metadata.set_source_line(*source_line);
  }
  if (profile_type) {
    for (const auto& type : *profile_type) {
      if (!ProfileType_IsValid(type)) {
        return false;
      }
      metadata.add_profile_type(static_cast<ProfileType>(type));
    }
  }
  if (deduplicated_name) {
    metadata.set_deduplicated_name(*deduplicated_name);
  }
  if (scheduling_name) {
    metadata.set_scheduling_name(*scheduling_name);
  }
  return true;
}

// ::= single_metadata | ('{' [single_metadata (',' single_metadata)*] '}')
bool HloParserImpl::ParseSingleOrListMetadata(
    std::vector<OpMetadata>& metadata) {
  if (lexer_.GetKind() == TokKind::kLbrace &&
      lexer_.LookAhead() == TokKind::kLbrace) {
    if (!ParseToken(TokKind::kLbrace, "expected '{' to start metadata list")) {
      return false;
    }

    if (lexer_.GetKind() != TokKind::kRbrace) {
      do {
        if (!ParseMetadata(metadata.emplace_back())) {
          return false;
        }
      } while (EatIfPresent(TokKind::kComma));
    }

    return ParseToken(TokKind::kRbrace, "expected '}' to end metadata list");
  }

  return ParseMetadata(metadata.emplace_back());
}

bool HloParserImpl::ParseOpShardingType(OpSharding::Type* type) {
  switch (lexer_.GetKind()) {
    case TokKind::kw_maximal:
      *type = OpSharding::MAXIMAL;
      lexer_.Lex();
      break;
    case TokKind::kw_replicated:
      *type = OpSharding::REPLICATED;
      lexer_.Lex();
      break;
    case TokKind::kw_manual:
      *type = OpSharding::MANUAL;
      lexer_.Lex();
      break;
    default:
      return false;
  }
  return true;
}

bool HloParserImpl::ParseListShardingType(
    std::vector<OpSharding::Type>* types) {
  if (!ParseToken(TokKind::kLbrace,
                  "expected '{' to start sharding type list")) {
    return false;
  }

  if (lexer_.GetKind() != TokKind::kRbrace) {
    do {
      OpSharding::Type type;
      if (!ParseOpShardingType(&type)) {
        return false;
      }
      types->push_back(type);
    } while (EatIfPresent(TokKind::kComma));
  }

  return ParseToken(TokKind::kRbrace, "expected '}' to end sharding type list");
}

bool HloParserImpl::ParseOpcode(
    HloOpcode* opcode, std::optional<HloOpcode>* async_wrapped_opcode) {
  VLOG(kDebugLevel) << "ParseOpcode";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects opcode");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToHloOpcode(val);
  if (!status_or_result.ok()) {
    auto try_parsing_async_op = [&](absl::string_view suffix,
                                    HloOpcode async_opcode) {
      absl::string_view wrapped_opcode_view(val);
      if (absl::ConsumeSuffix(&wrapped_opcode_view, suffix)) {
        *opcode = async_opcode;
        std::string wrapped_opcode(wrapped_opcode_view);
        status_or_result = StringToHloOpcode(wrapped_opcode);
        return true;
      }
      return false;
    };
    if (try_parsing_async_op("-start", HloOpcode::kAsyncStart) ||
        try_parsing_async_op("-update", HloOpcode::kAsyncUpdate) ||
        try_parsing_async_op("-done", HloOpcode::kAsyncDone)) {
      if (!status_or_result.ok()) {
        return TokenError(
            StrFormat("expects async wrapped opcode but sees: %s, error: %s",
                      val, status_or_result.status().message()));
      }
      *async_wrapped_opcode = status_or_result.value();
    } else {
      return TokenError(StrFormat("expects opcode but sees: %s, error: %s", val,
                                  status_or_result.status().message()));
    }
  } else {
    *opcode = status_or_result.value();
  }
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseFftType(FftType* result) {
  VLOG(kDebugLevel) << "ParseFftType";
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

bool HloParserImpl::ParsePaddingType(PaddingType* result) {
  VLOG(kDebugLevel) << "ParsePaddingType";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects padding type");
  }
  std::string val = lexer_.GetStrVal();
  if (!PaddingType_Parse(val, result) || !PaddingType_IsValid(*result)) {
    return TokenError(StrFormat("expects padding type but sees: %s", val));
  }
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseComparisonDirection(ComparisonDirection* result) {
  VLOG(kDebugLevel) << "ParseComparisonDirection";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects comparison direction");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToComparisonDirection(val);
  if (!status_or_result.ok()) {
    return TokenError(
        StrFormat("expects comparison direction but sees: %s", val));
  }
  *result = status_or_result.value();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseComparisonType(Comparison::Type* result) {
  VLOG(kDebugLevel) << "ParseComparisonType";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects comparison type");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToComparisonType(val);
  if (!status_or_result.ok()) {
    return TokenError(StrFormat("expects comparison type but sees: %s", val));
  }
  *result = status_or_result.value();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseFusionKind(HloInstruction::FusionKind* result) {
  VLOG(kDebugLevel) << "ParseFusionKind";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects fusion kind");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToFusionKind(val);
  if (!status_or_result.ok()) {
    return TokenError(StrFormat("expects fusion kind but sees: %s, error: %s",
                                val, status_or_result.status().message()));
  }
  *result = status_or_result.value();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseRandomDistribution(RandomDistribution* result) {
  VLOG(kDebugLevel) << "ParseRandomDistribution";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects random distribution");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToRandomDistribution(val);
  if (!status_or_result.ok()) {
    return TokenError(
        StrFormat("expects random distribution but sees: %s, error: %s", val,
                  status_or_result.status().message()));
  }
  *result = status_or_result.value();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseRandomAlgorithm(RandomAlgorithm* result) {
  VLOG(kDebugLevel) << "ParseRandomAlgorithm";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects random algorithm");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToRandomAlgorithm(val);
  if (!status_or_result.ok()) {
    return TokenError(
        StrFormat("expects random algorithm but sees: %s, error: %s", val,
                  status_or_result.status().message()));
  }
  *result = status_or_result.value();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParsePrecision(PrecisionConfig::Precision* result) {
  VLOG(kDebugLevel) << "ParsePrecision";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects random distribution");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToPrecision(val);
  if (!status_or_result.ok()) {
    return TokenError(StrFormat("expects precision but sees: %s, error: %s",
                                val, status_or_result.status().message()));
  }
  *result = status_or_result.value();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseAlgorithm(PrecisionConfig::Algorithm* result) {
  VLOG(kDebugLevel) << "ParseAlgorithm";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects algorithm");
  }
  std::string val = lexer_.GetStrVal();
  auto status_or_result = StringToAlgorithm(val);
  if (!status_or_result.ok()) {
    return TokenError(StrFormat("expects algorithm but sees: %s, error: %s",
                                val, status_or_result.status().message()));
  }
  *result = status_or_result.value();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseResultAccuracyType(ResultAccuracy::Mode* result) {
  VLOG(3) << "ParseResultAccuracyType";
  if (lexer_.GetKind() != TokKind::kIdent) {
    return TokenError("expects ResultAccuracy type");
  }
  std::string val = lexer_.GetStrVal();
  absl::StatusOr<ResultAccuracy::Mode> mode = StringToResultAccuracy(val);
  if (!mode.ok()) {
    return TokenError(
        StrFormat("expects ResultAccuracy type but sees: %s, error: %s", val,
                  mode.status().message()));
  }
  *result = mode.value();
  lexer_.Lex();
  return true;
}

bool HloParserImpl::ParseResultAccuracyTolerance(
    ResultAccuracy::Tolerance* result_tolerance) {
  VLOG(3) << "ParseResultAccuracyTolerance";
  if (!ParseToken(TokKind::kLbrace,
                  "expected '{' to start result accuracy list")) {
    return false;
  }
  double ulps = 0.0;
  double rtol = 0.0;
  double atol = 0.0;
  if (lexer_.GetKind() != TokKind::kRbrace) {
    do {
      std::string name;
      if (!ParseAttributeName(&name)) {
        return Error(lexer_.GetLoc(),
                     "expects string for result_accuracy tolerance type");
      }
      if (name == "ulps") {
        if (ParseDouble(&ulps)) {
          result_tolerance->set_ulps(ulps);
        }
      } else if (name == "rtol") {
        if (ParseDouble(&rtol)) {
          result_tolerance->set_rtol(rtol);
        }
      } else if (name == "atol") {
        if (ParseDouble(&atol)) {
          result_tolerance->set_atol(atol);  // NOLINT
        }
      } else {
        return Error(lexer_.GetLoc(),
                     StrFormat("invalid attribute name: %s", name));
      }
    } while (EatIfPresent(TokKind::kComma));
  }
  return ParseToken(TokKind::kRbrace,
                    "expects '}' at the end of result precision");
}

bool HloParserImpl::ParseResultAccuracy(ResultAccuracy* result) {
  VLOG(3) << "ParseResultAccuracy";
  if (!ParseToken(TokKind::kLbrace,
                  "expected '{' to start result precision list")) {
    return false;
  }
  ResultAccuracy::Mode mode;
  ResultAccuracy::Tolerance result_tolerance;
  std::string name;

  if (!ParseAttributeName(&name)) {
    return Error(lexer_.GetLoc(), "expects string for result_accuracy spec");
  }
  bool ok = [&] {
    if (name == "mode") {
      bool parse_mode = ParseResultAccuracyType(&mode);
      if (parse_mode) {
        result->set_mode(mode);
      }
      return parse_mode;
    }
    if (name == "tolerance") {
      bool parse_tolerance = ParseResultAccuracyTolerance(&result_tolerance);
      if (parse_tolerance) {
        *result->mutable_tolerance() = result_tolerance;
      }
      return parse_tolerance;
    }
    return Error(lexer_.GetLoc(),
                 StrFormat("invalid attribute name: %s", name));
  }();
  if (!ok) {
    return false;
  }
  return ParseToken(TokKind::kRbrace, "expected '}' to end result_accuracy");
}

bool HloParserImpl::ParseInt64(int64_t* result) {
  VLOG(kDebugLevel) << "ParseInt64";
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
  VLOG(kDebugLevel) << "ParseToken " << TokKindToString(kind) << " " << msg;
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

absl::StatusOr<Shape> HloParserImpl::ParseShapeOnly() {
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

absl::StatusOr<Layout> HloParserImpl::ParseLayoutOnly() {
  lexer_.Lex();
  Layout layout;
  if (!ParseLayout(&layout)) {
    return InvalidArgument("Syntax error:\n%s", GetError());
  }
  if (lexer_.GetKind() != TokKind::kEof) {
    return InvalidArgument("Syntax error:\nExtra content after layout");
  }
  return layout;
}

absl::StatusOr<HloSharding> HloParserImpl::ParseShardingOnly() {
  lexer_.Lex();
  std::optional<HloSharding> sharding;
  if (!ParseSharding(sharding)) {
    return InvalidArgument("Syntax error:\n%s", GetError());
  }
  if (lexer_.GetKind() != TokKind::kEof) {
    return InvalidArgument("Syntax error:\nExtra content after sharding");
  }
  return std::move(*sharding);
}

absl::StatusOr<FrontendAttributes>
HloParserImpl::ParseFrontendAttributesOnly() {
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

absl::StatusOr<StatisticsViz> HloParserImpl::ParseStatisticsVizOnly() {
  lexer_.Lex();
  StatisticsViz statistics_viz;
  if (!ParseStatisticsViz(&statistics_viz)) {
    return InvalidArgument("Syntax error:\n%s", GetError());
  }
  if (lexer_.GetKind() != TokKind::kEof) {
    return InvalidArgument("Syntax error:\nExtra content after statistics");
  }
  return statistics_viz;
}

absl::StatusOr<std::vector<bool>>
HloParserImpl::ParseParameterReplicationOnly() {
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

absl::StatusOr<HloParserImpl::BoolList>
HloParserImpl::ParseBooleanListOrSingleBooleanOnly() {
  lexer_.Lex();
  BoolList booleans;
  if (!ParseBooleanListOrSingleBoolean(&booleans)) {
    return InvalidArgument("Syntax error:\n%s", GetError());
  }
  if (lexer_.GetKind() != TokKind::kEof) {
    return InvalidArgument("Syntax error:\nExtra content after boolean list");
  }
  return booleans;
}

absl::StatusOr<std::vector<ReplicaGroup>>
HloParserImpl::ParseReplicaGroupsOnly() {
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

absl::StatusOr<CollectiveDeviceList>
HloParserImpl::ParseCollectiveDeviceListOnly() {
  lexer_.Lex();
  CollectiveDeviceList collective_device_list;
  if (!ParseCollectiveDeviceList(&collective_device_list)) {
    return InvalidArgument("Syntax error:\n%s", GetError());
  }
  if (lexer_.GetKind() != TokKind::kEof) {
    return InvalidArgument(
        "Syntax error:\nExtra content after collective device list");
  }
  return collective_device_list;
}

absl::StatusOr<Window> HloParserImpl::ParseWindowOnly() {
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

absl::StatusOr<ConvolutionDimensionNumbers>
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

absl::StatusOr<PaddingConfig> HloParserImpl::ParsePaddingConfigOnly() {
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
  int64_t parameter_count = 0;
  create_missing_instruction_ =
      [this, &builder, &parameter_count](
          const std::string& name,
          const Shape& shape) -> std::pair<HloInstruction*, LocTy>* {
    std::string new_name = name.empty() ? StrCat("_", parameter_count) : name;
    HloInstruction* parameter = builder.AddInstruction(
        HloInstruction::CreateParameter(parameter_count++, shape, new_name));
    current_name_table()[new_name] = {parameter, lexer_.GetLoc()};
    return tsl::gtl::FindOrNull(current_name_table(), new_name);
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
        "Syntax error:\nExpected eof after parsing single instruction. Did you"
        " mean to write an HLO module and forget the \"HloModule\" header?");
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

absl::StatusOr<std::unique_ptr<HloModule>> ParseAndReturnUnverifiedModule(
    absl::string_view str, const HloModuleConfig& config,
    const HloParserOptions& options) {
  auto module = std::make_unique<HloModule>(/*name=*/"_", config);
  HloParserImpl parser(str, options);
  TF_RETURN_IF_ERROR(parser.Run(module.get()));
  return module;
}

absl::StatusOr<HloSharding> ParseSharding(absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseShardingOnly();
}

absl::StatusOr<FrontendAttributes> ParseFrontendAttributes(
    absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseFrontendAttributesOnly();
}

absl::StatusOr<StatisticsViz> ParseStatisticsViz(absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseStatisticsVizOnly();
}

absl::StatusOr<std::vector<bool>> ParseParameterReplication(
    absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseParameterReplicationOnly();
}

absl::StatusOr<HloParserImpl::BoolList> ParseBooleanListOrSingleBoolean(
    absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseBooleanListOrSingleBooleanOnly();
}

absl::StatusOr<std::vector<ReplicaGroup>> ParseReplicaGroupsOnly(
    absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseReplicaGroupsOnly();
}

absl::StatusOr<CollectiveDeviceList> ParseCollectiveDeviceListOnly(
    absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseCollectiveDeviceListOnly();
}

absl::StatusOr<Window> ParseWindow(absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseWindowOnly();
}

absl::StatusOr<ConvolutionDimensionNumbers> ParseConvolutionDimensionNumbers(
    absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseConvolutionDimensionNumbersOnly();
}

absl::StatusOr<PaddingConfig> ParsePaddingConfig(absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParsePaddingConfigOnly();
}

absl::StatusOr<Shape> ParseShape(absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseShapeOnly();
}

absl::StatusOr<Layout> ParseLayout(absl::string_view str) {
  HloParserImpl parser(str);
  return parser.ParseLayoutOnly();
}

std::unique_ptr<HloParser> HloParser::CreateHloParserForTests(
    absl::string_view str, const HloParserOptions& options) {
  return std::make_unique<HloParserImpl>(str, options);
}

}  // namespace xla
