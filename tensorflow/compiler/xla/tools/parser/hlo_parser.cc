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

#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace xla {
namespace tools {

namespace {

using tensorflow::StringPiece;
using tensorflow::strings::Printf;
using tensorflow::strings::StrAppend;
using tensorflow::strings::StrCat;

const double kF16max = 65504;

// Parser for the HloModule::ToString() format text.
class HloParser {
 public:
  explicit HloParser(StringPiece str, const HloModuleConfig& config)
      : lexer_(str), config_(config) {}

  // Runs the parser. Returns false if an error occurred.
  bool Run();

  // Returns the parsed HloModule.
  std::unique_ptr<HloModule> ConsumeHloModule() { return std::move(module_); }

  // Returns the error information.
  string GetError() const { return tensorflow::str_util::Join(error_, "\n"); }

 private:
  // ParseXXX returns false if an error occurred.
  bool ParseHloModule();
  bool ParseComputations();
  bool ParseComputation();
  bool ParseInstructionList(HloComputation::Builder* builder,
                            string* root_name);
  bool ParseInstruction(HloComputation::Builder* builder, string* root_name);
  bool ParseSharding(HloInstruction* instruction);
  bool ParseLiteral(std::unique_ptr<Literal>* literal, const Shape& shape);
  bool ParseTupleLiteral(std::unique_ptr<Literal>* literal, const Shape& shape);
  bool ParseNonTupleLiteral(std::unique_ptr<Literal>* literal,
                            const Shape& shape);
  // Sets the sub-value of literal at the given index to the given value. The
  // literal's shape must have the default layout.
  bool SetValueInLiteral(int64 value, int64 linear_index, Literal* literal);
  bool SetValueInLiteral(double value, int64 linear_index, Literal* literal);
  bool SetValueInLiteral(bool value, int64 linear_index, Literal* literal);
  template <typename LiteralNativeT, typename ParsedElemT>
  bool SetValueInLiteralHelper(ParsedElemT value, int64 linear_index,
                               Literal* literal);

  bool ParseOperands(std::vector<HloInstruction*>* operands);
  // Fills parsed operands into 'operands' and expects a certain number of
  // operands.
  bool ParseOperands(std::vector<HloInstruction*>* operands,
                     const int expected_size);

  template <typename T>
  bool ParseExtraAttribute(T* value, const string& expected_attribute);
  template <typename T>
  bool ParseAttributeValue(T* value);

  bool ParseParamList();
  bool ParseName(string* result);
  bool ParseAttributeName(string* result);
  bool ParseShape(Shape* result);
  bool ParseOpcode(HloOpcode* result);
  bool ParseInt64(int64* result);
  bool ParseDouble(double* result);
  bool ParseBool(bool* result);
  bool ParseToken(TokKind kind, const string& msg);

  // Logs the current parsing line and the given message. Always returns false.
  bool TokenError(StringPiece msg);

  // If the current token is 'kind', eats it (i.e. lexes the next token) and
  // returns true.
  bool EatIfPresent(TokKind kind);
  // Parses a shape, and returns true if the result is compatible with the given
  // shape.
  bool EatShapeAndCheckCompatible(const Shape& shape);

  // Adds the instruction to the pool. Returns false and emits an error if the
  // instruction already exists.
  bool AddInstruction(const string& name, HloInstruction* instruction);
  // Adds the computation to the pool. Returns false and emits an error if the
  // computation already exists.
  bool AddComputation(const string& name, HloComputation* computation);

  // The map from the instruction name to the instruction. This does not own the
  // instructions.
  std::unordered_map<string, HloInstruction*> instruction_pool_;
  std::unordered_map<string, HloComputation*> computation_pool_;

  HloLexer lexer_;
  std::unique_ptr<HloModule> module_;
  const HloModuleConfig config_;
  std::vector<string> error_;
};

bool HloParser::TokenError(StringPiece msg) {
  const string error =
      StrCat("was parsing \"", lexer_.GetCurrentLine(), "\"; token ",
             TokKindToString(lexer_.GetKind()), "; ", msg);
  VLOG(1) << "TokenError: " << error;
  error_.push_back(error);
  return false;
}

bool HloParser::Run() {
  lexer_.Lex();
  return ParseHloModule();
}

// ::= 'HloModule' name computations
bool HloParser::ParseHloModule() {
  if (lexer_.GetKind() != TokKind::kw_HloModule) {
    return TokenError("expects HloModule");
  }
  // Eat 'HloModule'
  lexer_.Lex();

  string name;
  if (!ParseName(&name)) {
    return false;
  }

  module_ = MakeUnique<HloModule>(name, config_);

  return ParseComputations();
}

// computations ::= (computation)+
bool HloParser::ParseComputations() {
  do {
    if (!ParseComputation()) {
      return false;
    }
  } while (lexer_.GetKind() != TokKind::kEof);
  return true;
}

// computation ::= ('ENTRY')? name param_list '->' shape instruction_list
bool HloParser::ParseComputation() {
  const bool is_entry_computation = EatIfPresent(TokKind::kw_ENTRY);
  string name;
  if (!ParseName(&name)) {
    return false;
  }
  auto builder = MakeUnique<HloComputation::Builder>(name);

  Shape shape;
  string root_name;
  if (!ParseParamList() || !ParseToken(TokKind::kArrow, "expects '->'") ||
      !ParseShape(&shape) || !ParseInstructionList(builder.get(), &root_name)) {
    return false;
  }

  HloInstruction* root =
      tensorflow::gtl::FindPtrOrNull(instruction_pool_, root_name);
  // This means some instruction was marked as ROOT but we didn't find it in the
  // pool, which should not happen.
  if (!root_name.empty() && root == nullptr) {
    LOG(FATAL) << "instruction " << root_name
               << " was marked as ROOT but the parser has not seen it before";
  }
  // Now root can be either an existing instruction or a nullptr. If it's a
  // nullptr, the implementation of Builder will set the last instruction as
  // root instruction.
  HloComputation* computation =
      is_entry_computation
          ? module_->AddEntryComputation(builder->Build(root))
          : module_->AddEmbeddedComputation(builder->Build(root));
  return AddComputation(name, computation);
}

// instruction_list ::= '{' instruction_list1 '}'
// instruction_list1 ::= (instruction)+
bool HloParser::ParseInstructionList(HloComputation::Builder* builder,
                                     string* root_name) {
  if (!ParseToken(TokKind::kLbrace,
                  "expects '{' at the beginning of instruction list.")) {
    return false;
  }
  do {
    if (!ParseInstruction(builder, root_name)) {
      return false;
    }
  } while (lexer_.GetKind() != TokKind::kRbrace);
  return ParseToken(TokKind::kRbrace,
                    "expects '}' at the end of instruction list.");
}

// instruction ::= ('ROOT')? name '=' shape opcode operands (extra_attribute)*
bool HloParser::ParseInstruction(HloComputation::Builder* builder,
                                 string* root_name) {
  string name;
  Shape shape;
  HloOpcode opcode;
  std::vector<HloInstruction*> operands;
  bool is_root = EatIfPresent(TokKind::kw_ROOT);
  if (!ParseName(&name) ||
      !ParseToken(TokKind::kEqual, "expects '=' in instruction") ||
      !ParseShape(&shape) || !ParseOpcode(&opcode)) {
    return false;
  }
  if (is_root) {
    *root_name = name;
  }
  HloInstruction* instruction;
  switch (opcode) {
    case HloOpcode::kParameter: {
      int64 parameter_number;
      if (!ParseToken(TokKind::kLparen,
                      "expects '(' before parameter number") ||
          !ParseInt64(&parameter_number) ||
          !ParseToken(TokKind::kRparen, "expects ')' after parameter number")) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateParameter(parameter_number, shape, name));
      break;
    }
    case HloOpcode::kConstant: {
      std::unique_ptr<Literal> literal;
      if (!ParseToken(TokKind::kLparen,
                      "expects '(' before constant literal") ||
          !ParseLiteral(&literal, shape) ||
          !ParseToken(TokKind::kRparen, "expects ')' after constant literal")) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateConstant(std::move(literal)));
      break;
    }
    // Unary ops.
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kReal:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSort:
    case HloOpcode::kTanh: {
      if (!ParseOperands(&operands, /*expected_size=*/1)) {
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
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kNe:
    case HloOpcode::kDot:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical: {
      if (!ParseOperands(&operands, /*expected_size=*/2)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateBinary(
          shape, opcode, operands[0], operands[1]));
      break;
    }
    // Ternary ops.
    case HloOpcode::kClamp:
    case HloOpcode::kSelect: {
      if (!ParseOperands(&operands, /*expected_size=*/3)) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateTernary(
          shape, opcode, operands[0], operands[1], operands[2]));
      break;
    }
    // Other supported ops.
    case HloOpcode::kConvert: {
      if (!ParseOperands(&operands, /*expected_size=*/1)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateConvert(shape, operands[0]));
      break;
    }
    case HloOpcode::kCrossReplicaSum: {
      if (!ParseOperands(&operands, /*expected_size=*/1)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateCrossReplicaSum(shape, operands[0]));
      break;
    }
    case HloOpcode::kReshape: {
      if (!ParseOperands(&operands, /*expected_size=*/1)) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateReshape(shape, operands[0]));
      break;
    }
    case HloOpcode::kTuple: {
      if (!ParseOperands(&operands)) {
        return false;
      }
      instruction =
          builder->AddInstruction(HloInstruction::CreateTuple(operands));
      break;
    }
    case HloOpcode::kWhile: {
      HloComputation* condition;
      HloComputation* body;
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseExtraAttribute(&condition,
                               /*expected_attribute=*/"condition") ||
          !ParseExtraAttribute(&body, /*expected_attribute=*/"body")) {
        return false;
      }
      instruction = builder->AddInstruction(HloInstruction::CreateWhile(
          shape, condition, body, /*init=*/operands[0]));
      break;
    }
    case HloOpcode::kRecv: {
      int64 channel_id;
      if (!ParseOperands(&operands, /*expected_size=*/0) ||
          !ParseExtraAttribute(&channel_id,
                               /*expected_attribute=*/"channel_id")) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateRecv(shape, channel_id));
      break;
    }
    case HloOpcode::kSend: {
      int64 channel_id;
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseExtraAttribute(&channel_id,
                               /*expected_attribute=*/"channel_id")) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateSend(operands[0], channel_id));
      break;
    }
    case HloOpcode::kGetTupleElement: {
      int64 index;
      if (!ParseOperands(&operands, /*expected_size=*/1) ||
          !ParseExtraAttribute(&index, /*expected_attribute=*/"index")) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateGetTupleElement(shape, operands[0], index));
      break;
    }
    case HloOpcode::kCall: {
      HloComputation* to_apply;
      if (!ParseOperands(&operands) ||
          !ParseExtraAttribute(&to_apply,
                               /*expected_attribute=*/"to_apply")) {
        return false;
      }
      instruction = builder->AddInstruction(
          HloInstruction::CreateCall(shape, operands, to_apply));
      break;
    }
    case HloOpcode::kBroadcast:
    case HloOpcode::kCustomCall:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kConvolution:
    case HloOpcode::kMap:
    case HloOpcode::kPad:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kReverse:
    case HloOpcode::kRng:
    case HloOpcode::kSlice:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kFusion:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kTrace:
      return TokenError(StrCat("parsing not yet implemented for op: ",
                               HloOpcodeString(opcode)));
  }
  // Parse "sharding=".
  if (lexer_.GetKind() == TokKind::kComma) {
    if (!ParseSharding(instruction)) {
      return false;
    }
  }

  return AddInstruction(name, instruction);
}

// ::= '{' 'replicated'? 'maximal'? ('device=' int)? shape? ('devices=' ('['
// dims ']')* device_list)? '}' dims ::= int_list device_list ::= int_list
bool HloParser::ParseSharding(HloInstruction* instruction) {
  if (!ParseToken(TokKind::kComma,
                  "expects ',' in front of an extra attribute")) {
    return false;
  }
  string attribute_name;
  if (!ParseAttributeName(&attribute_name) || attribute_name != "sharding") {
    return TokenError("expects attribute name: sharding");
  }

  if (!ParseToken(TokKind::kLbrace,
                  "expected '{' to start sharding attribute")) {
    return false;
  }

  bool maximal = false;
  bool replicated = false;
  std::vector<int64> devices;
  std::vector<int64> tile_assignment_dimensions;
  Shape tile_shape;
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
      case TokKind::kShape:
        tile_shape = lexer_.GetShapeVal();
        lexer_.Lex();
        break;
      case TokKind::kRbrace:
        break;
      default:
        return TokenError("unexpected token");
    }
  }

  OpSharding sharding;
  if (replicated) {
    if (!devices.empty()) {
      return TokenError(
          "replicated shardings should not have any devices assigned");
    }
    if (!ShapeUtil::Equal(tile_shape, Shape())) {
      return TokenError(
          "replicated shardings should not have any tile shape set");
    }
    sharding.set_type(OpSharding::Type::OpSharding_Type_REPLICATED);
  } else if (maximal) {
    if (devices.size() != 1) {
      return TokenError(
          "maximal shardings should have exactly one device assigned");
    }
    if (!ShapeUtil::Equal(tile_shape, Shape())) {
      return TokenError("maximal shardings should not have any tile shape set");
    }
    sharding.set_type(OpSharding::Type::OpSharding_Type_MAXIMAL);
    sharding.add_tile_assignment_devices(devices[0]);
  } else {
    if (devices.size() <= 1) {
      return TokenError(
          "non-maximal shardings must have more than one device assigned");
    }
    if (ShapeUtil::Equal(tile_shape, Shape())) {
      return TokenError("non-maximal shardings should have a tile shape set");
    }
    if (tile_assignment_dimensions.empty()) {
      return TokenError(
          "non-maximal shardings must have a tile assignment list including "
          "dimensions");
    }
    sharding.set_type(OpSharding::Type::OpSharding_Type_OTHER);
    *sharding.mutable_tile_shape() = tile_shape;
    for (int64 dim : tile_assignment_dimensions) {
      sharding.add_tile_assignment_dimensions(dim);
    }
    for (int64 device : devices) {
      sharding.add_tile_assignment_devices(device);
    }
  }

  instruction->set_sharding(HloSharding::FromProto(sharding).ValueOrDie());
  lexer_.Lex();
  return true;
}

bool HloParser::SetValueInLiteral(int64 value, int64 linear_index,
                                  Literal* literal) {
  const Shape& shape = literal->shape();
  switch (shape.element_type()) {
    case S8:
      return SetValueInLiteralHelper<int8>(value, linear_index, literal);
    case S16:
      return SetValueInLiteralHelper<int16>(value, linear_index, literal);
    case S32:
      return SetValueInLiteralHelper<int32>(value, linear_index, literal);
    case S64:
      return SetValueInLiteralHelper<int64>(value, linear_index, literal);
    case U8:
      return SetValueInLiteralHelper<uint8>(value, linear_index, literal);
    case U16:
      return SetValueInLiteralHelper<uint8>(value, linear_index, literal);
    case U32:
      return SetValueInLiteralHelper<uint32>(value, linear_index, literal);
    case U64:
      return SetValueInLiteralHelper<uint64>(value, linear_index, literal);
    default:
      LOG(FATAL) << "unknown integral primitive type "
                 << PrimitiveType_Name(shape.element_type());
  }
}

bool HloParser::SetValueInLiteral(double value, int64 linear_index,
                                  Literal* literal) {
  const Shape& shape = literal->shape();
  switch (shape.element_type()) {
    case F16:
      return SetValueInLiteralHelper<half>(value, linear_index, literal);
    case F32:
      return SetValueInLiteralHelper<float>(value, linear_index, literal);
    case F64:
      return SetValueInLiteralHelper<double>(value, linear_index, literal);
    default:
      LOG(FATAL) << "unknown floating point primitive type "
                 << PrimitiveType_Name(shape.element_type());
  }
}

bool HloParser::SetValueInLiteral(bool value, int64 linear_index,
                                  Literal* literal) {
  const Shape& shape = literal->shape();
  switch (shape.element_type()) {
    case PRED:
      return SetValueInLiteralHelper<bool>(value, linear_index, literal);
    default:
      LOG(FATAL) << PrimitiveType_Name(shape.element_type())
                 << " is not PRED type";
  }
}

template <typename LiteralNativeT, typename ParsedElemT>
bool HloParser::SetValueInLiteralHelper(ParsedElemT value, int64 linear_index,
                                        Literal* literal) {
  // Check that linear_index is in range.
  if (linear_index >= ShapeUtil::ElementsIn(literal->shape())) {
    return TokenError(
        StrCat("trys to set value ", value, " to a literal in shape ",
               ShapeUtil::HumanString(literal->shape()), " at linear index ",
               linear_index, ", but the index is out of range"));
  }

  if (std::isnan(value) ||
      (std::numeric_limits<ParsedElemT>::has_infinity &&
       (std::numeric_limits<ParsedElemT>::infinity() == value ||
        -std::numeric_limits<ParsedElemT>::infinity() == value))) {
    // Skip range checking for non-finite value.
  } else if (literal->shape().element_type() == F16) {
    if (value > kF16max || value < -kF16max) {
      return TokenError(StrCat(
          "value ", value, " is out of range for literal's primitive type ",
          PrimitiveType_Name(literal->shape().element_type())));
    }
  } else if (value > static_cast<ParsedElemT>(
                         std::numeric_limits<LiteralNativeT>::max()) ||
             value < static_cast<ParsedElemT>(
                         std::numeric_limits<LiteralNativeT>::lowest())) {
    // Value is out of range for LiteralNativeT.
    return TokenError(StrCat(
        "value ", value, " is out of range for literal's primitive type ",
        PrimitiveType_Name(literal->shape().element_type())));
  }

  literal->GetMutableArraySlice<LiteralNativeT>().at(linear_index) =
      static_cast<LiteralNativeT>(value);
  return true;
}

bool HloParser::EatShapeAndCheckCompatible(const Shape& shape) {
  Shape new_shape;
  if (!ParseShape(&new_shape)) {
    return TokenError(StrCat("expects shape ", ShapeUtil::HumanString(shape)));
  }
  if (!ShapeUtil::Compatible(shape, new_shape)) {
    return TokenError(StrCat(
        "expects shape ", ShapeUtil::HumanString(shape),
        ", but sees a different shape: ", ShapeUtil::HumanString(new_shape)));
  }
  return true;
}

// literal
//  ::= tuple
//  ::= non_tuple
bool HloParser::ParseLiteral(std::unique_ptr<Literal>* literal,
                             const Shape& shape) {
  return ShapeUtil::IsTuple(shape) ? ParseTupleLiteral(literal, shape)
                                   : ParseNonTupleLiteral(literal, shape);
}

// tuple
//  ::= shape '(' literal_list ')'
// literal_list
//  ::= /*empty*/
//  ::= literal (',' literal)*
bool HloParser::ParseTupleLiteral(std::unique_ptr<Literal>* literal,
                                  const Shape& shape) {
  if (!EatShapeAndCheckCompatible(shape)) {
    return TokenError(StrCat("expects tuple constant in shape ",
                             ShapeUtil::HumanString(shape)));
  }
  if (!ParseToken(TokKind::kLparen, "expects '(' in front of tuple elements")) {
    return false;
  }
  std::vector<std::unique_ptr<Literal>> elements(
      ShapeUtil::TupleElementCount(shape));

  if (lexer_.GetKind() == TokKind::kRparen) {
    // empty
  } else {
    // literal, (',' literal)*
    for (int i = 0; i < elements.size(); i++) {
      if (i > 0) {
        ParseToken(TokKind::kComma, "exepcts ',' to separate tuple elements");
      }
      if (!ParseLiteral(&elements[i],
                        ShapeUtil::GetTupleElementShape(shape, i))) {
        return TokenError(StrCat("expects the ", i, "th element"));
      }
    }
  }
  *literal = Literal::MakeTupleOwned(std::move(elements));
  return ParseToken(TokKind::kRparen,
                    StrCat("expects ')' at the end of the tuple with ",
                           ShapeUtil::TupleElementCount(shape), "elements"));
}

// non_tuple
//   ::= rank01
//   ::= rank2345
// rank2345 ::= shape nested_array
bool HloParser::ParseNonTupleLiteral(std::unique_ptr<Literal>* literal,
                                     const Shape& shape) {
  const int64 size = ShapeUtil::ElementsIn(shape);
  if (size == 0) {
    *literal = Literal::CreateFromShape(shape);
    return true;
  }

  const int64 rank = ShapeUtil::Rank(shape);
  if (rank > 1 && !EatShapeAndCheckCompatible(shape)) {
    return false;
  }

  // Create a literal with the given shape in default layout.
  *literal = Literal::CreateFromDimensions(shape.element_type(),
                                           AsInt64Slice(shape.dimensions()));
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
  auto get_index_str = [&elems_seen_per_dim](int dim) -> string {
    std::vector<int64> elems_seen_until_dim(elems_seen_per_dim.begin(),
                                            elems_seen_per_dim.begin() + dim);
    return StrCat("[",
                  tensorflow::str_util::Join(
                      elems_seen_until_dim, ",",
                      [](string* out, const int64& num_elems) {
                        tensorflow::strings::StrAppend(out, num_elems - 1);
                      }),
                  "]");
  };
  do {
    switch (lexer_.GetKind()) {
      default:
        return TokenError("unexpected token type in a literal");
      case TokKind::kLbrace: {
        nest_level++;
        if (nest_level > rank) {
          return TokenError(Printf(
              "expects nested array in rank %lld, but sees larger", rank));
        }
        if (nest_level > 1) {
          elems_seen_per_dim[nest_level - 2]++;
          if (elems_seen_per_dim[nest_level - 2] >
              shape.dimensions(nest_level - 2)) {
            return TokenError(Printf(
                "expects %lld elements in the %sth element, but sees more",
                shape.dimensions(nest_level - 2),
                get_index_str(nest_level - 2).c_str()));
          }
        }
        lexer_.Lex();
        break;
      }
      case TokKind::kRbrace: {
        nest_level--;
        if (elems_seen_per_dim[nest_level] != shape.dimensions(nest_level)) {
          return TokenError(Printf(
              "expects %lld elements in the %sth element, but sees %lld",
              shape.dimensions(nest_level), get_index_str(nest_level).c_str(),
              elems_seen_per_dim[nest_level]));
        }
        elems_seen_per_dim[nest_level] = 0;
        lexer_.Lex();
        break;
      }
      case TokKind::kComma:
      case TokKind::kComment:
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
        if (rank > 0) {
          if (nest_level != rank) {
            return TokenError(
                Printf("expects nested array in rank %lld, but sees %lld", rank,
                       nest_level));
          }
          elems_seen_per_dim[rank - 1]++;
          if (elems_seen_per_dim[rank - 1] > shape.dimensions(rank - 1)) {
            return TokenError(
                Printf("expects %lld elements on the minor-most dimension, but "
                       "sees more",
                       shape.dimensions(rank - 1)));
          }
        }
        if (lexer_.GetKind() == TokKind::kw_true ||
            lexer_.GetKind() == TokKind::kw_false) {
          // TODO(congliu): bool type literals with rank >= 1 are actually
          // printed in a compact form instead of "true" or "false". Fix that.
          if (!SetValueInLiteral(lexer_.GetKind() == TokKind::kw_true,
                                 linear_index++, literal->get())) {
            return false;
          }
          lexer_.Lex();
        } else if (primitive_util::IsIntegralType(shape.element_type())) {
          int64 value;
          if (!ParseInt64(&value)) {
            return TokenError(StrCat("expects integer for primitive type: ",
                                     PrimitiveType_Name(shape.element_type())));
          }
          if (!SetValueInLiteral(value, linear_index++, literal->get())) {
            return false;
          }
        } else if (primitive_util::IsFloatingPointType(shape.element_type())) {
          double value;
          if (!ParseDouble(&value)) {
            return TokenError(
                StrCat("expect floating point value for primitive type: ",
                       PrimitiveType_Name(shape.element_type())));
          }
          if (!SetValueInLiteral(value, linear_index++, literal->get())) {
            return false;
          }
        } else {
          return TokenError(StrCat("unsupported premitive type ",
                                   PrimitiveType_Name(shape.element_type())));
        }
        break;
      }
    }  // end of switch
  } while (nest_level > 0);

  *literal = (*literal)->Relayout(shape.layout());
  return true;
}

// operands ::= '(' operands1 ')'
// operands1
//   ::= /*empty*/
//   ::= operand (, operand)*
// operand ::= shape name
bool HloParser::ParseOperands(std::vector<HloInstruction*>* operands) {
  if (!ParseToken(TokKind::kLparen,
                  "expects '(' at the beginning of operands")) {
    return false;
  }
  if (lexer_.GetKind() == TokKind::kRparen) {
    // empty
  } else {
    do {
      Shape shape;
      string name;
      if (!ParseShape(&shape) || !ParseName(&name)) {
        return false;
      }
      HloInstruction* instruction =
          tensorflow::gtl::FindPtrOrNull(instruction_pool_, name);
      if (!instruction) {
        return TokenError(StrCat("instruction does not exist: ", name));
      }
      operands->push_back(instruction);
    } while (EatIfPresent(TokKind::kComma));
  }
  return ParseToken(TokKind::kRparen, "expects ')' at the end of operands");
}

bool HloParser::ParseOperands(std::vector<HloInstruction*>* operands,
                              const int expected_size) {
  if (!ParseOperands(operands)) {
    return false;
  }
  if (expected_size != operands->size()) {
    return TokenError(StrCat("expects ", expected_size, " operands, but has ",
                             operands->size(), " operands"));
  }
  return true;
}

// extra_attribute ::= ',' attribute_name value
template <typename T>
bool HloParser::ParseExtraAttribute(T* value,
                                    const string& expected_attribute) {
  if (!ParseToken(TokKind::kComma,
                  "expects ',' in front of an extra attribute")) {
    return false;
  }
  string attribute_name;
  if (!ParseAttributeName(&attribute_name) &&
      attribute_name != expected_attribute) {
    return TokenError(StrCat("expects attribute name: ", expected_attribute));
  }
  if (!ParseAttributeValue(value)) {
    return TokenError(
        StrCat("expects value for attribute: ", expected_attribute));
  }
  return true;
}

template <>
bool HloParser::ParseAttributeValue<HloComputation*>(HloComputation** value) {
  string name;
  if (!ParseName(&name)) {
    return TokenError("expects computation name");
  }
  *value = tensorflow::gtl::FindPtrOrNull(computation_pool_, name);
  if (*value == nullptr) {
    return TokenError(StrCat("computation does not exist: ", name));
  }
  return true;
}

template <>
bool HloParser::ParseAttributeValue<int64>(int64* value) {
  return ParseInt64(value);
}

// param_list ::= '(' param_list1 ')'
// param_list1
//   ::= /*empty*/
//   ::= param (',' param)*
// param ::= name shape
bool HloParser::ParseParamList() {
  if (!ParseToken(TokKind::kLparen,
                  "expects '(' at the beginning of param list")) {
    return false;
  }

  if (lexer_.GetKind() == TokKind::kRparen) {
    // empty
  } else {
    do {
      Shape shape;
      if (!ParseToken(TokKind::kName, "expects name in parameter") ||
          !ParseShape(&shape)) {
        return false;
      }
    } while (EatIfPresent(TokKind::kComma));
  }
  return ParseToken(TokKind::kRparen, "expects ')' at the end of param list");
}

// shape ::= shape_val_
// shape ::= '(' tuple_elements ')'
// tuple_elements
//   ::= /*empty*/
//   ::= shape (',' shape)*
bool HloParser::ParseShape(Shape* result) {
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

  if (lexer_.GetKind() != TokKind::kShape) {
    return TokenError("expects shape");
  }
  *result = lexer_.GetShapeVal();
  lexer_.Lex();
  return true;
}

bool HloParser::ParseName(string* result) {
  VLOG(1) << "ParseName";
  if (lexer_.GetKind() != TokKind::kName) {
    return TokenError("expects name");
  }
  *result = lexer_.GetStrVal();
  lexer_.Lex();
  return true;
}

bool HloParser::ParseAttributeName(string* result) {
  if (lexer_.GetKind() != TokKind::kAttributeName) {
    return TokenError("expects attribute name");
  }
  *result = lexer_.GetStrVal();
  lexer_.Lex();
  return true;
}

bool HloParser::ParseOpcode(HloOpcode* result) {
  VLOG(1) << "ParseOpcode";
  if (lexer_.GetKind() != TokKind::kOpcode) {
    return TokenError("expects opcode");
  }
  *result = lexer_.GetOpcodeVal();
  lexer_.Lex();
  return true;
}

bool HloParser::ParseInt64(int64* result) {
  VLOG(1) << "ParseInt64";
  if (lexer_.GetKind() != TokKind::kInt) {
    return TokenError("expects integer");
  }
  *result = lexer_.GetInt64Val();
  lexer_.Lex();
  return true;
}

bool HloParser::ParseDouble(double* result) {
  switch (lexer_.GetKind()) {
    case TokKind::kDecimal:
      *result = lexer_.GetDecimalVal();
      break;
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

bool HloParser::ParseBool(bool* result) {
  if (lexer_.GetKind() != TokKind::kw_true &&
      lexer_.GetKind() != TokKind::kw_false) {
    return TokenError("expects true or false");
  }
  *result = lexer_.GetKind() == TokKind::kw_true;
  lexer_.Lex();
  return true;
}

bool HloParser::ParseToken(TokKind kind, const string& msg) {
  VLOG(1) << "ParseToken " << TokKindToString(kind) << " " << msg;
  if (lexer_.GetKind() != kind) {
    return TokenError(msg);
  }
  lexer_.Lex();
  return true;
}

bool HloParser::EatIfPresent(TokKind kind) {
  if (lexer_.GetKind() != kind) {
    return false;
  }
  lexer_.Lex();
  return true;
}

bool HloParser::AddInstruction(const string& name,
                               HloInstruction* instruction) {
  auto result = instruction_pool_.insert({name, instruction});
  if (!result.second) {
    return TokenError(StrCat("instruction already exists: ", name));
  }
  return true;
}

bool HloParser::AddComputation(const string& name,
                               HloComputation* computation) {
  auto result = computation_pool_.insert({name, computation});
  if (!result.second) {
    return TokenError(StrCat("computation already exists: ", name));
  }
  return true;
}

}  // namespace

StatusOr<std::unique_ptr<HloModule>> Parse(StringPiece str,
                                           const HloModuleConfig& config) {
  HloParser parser(str, config);
  if (!parser.Run()) {
    return InvalidArgument("Syntax error: %s", parser.GetError().c_str());
  }
  return parser.ConsumeHloModule();
}

StatusOr<std::unique_ptr<HloModule>> Parse(StringPiece str) {
  HloModuleConfig config;
  return Parse(str, config);
}

}  // namespace tools
}  // namespace xla
