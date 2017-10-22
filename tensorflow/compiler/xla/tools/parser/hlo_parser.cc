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

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {
namespace tools {

namespace {

using tensorflow::StringPiece;
using tensorflow::strings::StrCat;

// Parser for the HloModule::ToString() format text.
class HloParser {
 public:
  explicit HloParser(StringPiece str) : lexer_(str) {}

  // Runs the parser. Returns false if an error occurred.
  bool Run();

  // Returns the parsed HloModule.
  std::unique_ptr<HloModule> ConsumeHloModule() { return std::move(module_); }

  // Returns the error information.
  string GetError() const { return tensorflow::str_util::Join(error_, "\n"); }

 private:
  // ParseXXX returns false if an error occurred.
  bool ParseHloModule();
  bool ParseComputation();
  bool ParseInstructionList(HloComputation::Builder* builder);
  bool ParseInstruction(HloComputation::Builder* builder);
  bool ParseLiteral(std::unique_ptr<Literal>* literal, const Shape& shape);
  bool ParseOperands(std::vector<HloInstruction*>* operands,
                     const int expected_size);
  bool ParseParamList();
  bool ParseName(string* result);
  bool ParseShape(Shape* result);
  bool ParseOpcode(HloOpcode* result);
  bool ParseInt64(int64* result);
  bool ParseDecimal(double* result);
  bool ParseBool(bool* result);
  bool ParseToken(TokKind kind, const string& msg);

  // Logs the current parsing line and the given message. Always returns false.
  bool TokenError(StringPiece msg);

  // If the current token is 'kind', eats it (i.e. lexes the next token) and
  // returns true.
  bool EatIfPresent(TokKind kind);

  // Adds the instruction to the pool. Returns false and emits an error if the
  // instruction already exists.
  bool AddInstruction(const string& name, HloInstruction* instruction);

  // The map from the instruction name to the instruction. This does not own the
  // instructions.
  std::unordered_map<string, HloInstruction*> instruction_pool_;

  HloLexer lexer_;
  std::unique_ptr<HloModule> module_;
  std::vector<string> error_;
};

bool HloParser::TokenError(StringPiece msg) {
  error_.push_back(
      StrCat("was parsing \"", lexer_.GetCurrentLine(), "\"; ", msg));
  return false;
}

bool HloParser::Run() {
  lexer_.Lex();
  return ParseHloModule();
}

// ::= 'HloModule' name computation
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

  module_ = MakeUnique<HloModule>(name);

  return ParseComputation();
}

// computation ::= 'ENTRY' name param_list '->' shape instruction_list
bool HloParser::ParseComputation() {
  string name;
  if (!ParseToken(TokKind::kw_ENTRY, "expects 'ENTRY'") || !ParseName(&name)) {
    return false;
  }
  auto builder = MakeUnique<HloComputation::Builder>(name);

  Shape shape;
  if (!ParseParamList() || !ParseToken(TokKind::kArrow, "expects '->'") ||
      !ParseShape(&shape) || !ParseInstructionList(builder.get())) {
    return false;
  }
  module_->AddEntryComputation(builder->Build());
  return true;
}

// instruction_list ::= '{' instruction_list1 '}'
// instruction_list1 ::= (instruction)+
bool HloParser::ParseInstructionList(HloComputation::Builder* builder) {
  if (!ParseToken(TokKind::kLbrace,
                  "expects '{' at the beginning of instruction list.")) {
    return false;
  }
  do {
    if (!ParseInstruction(builder)) {
      return false;
    }
  } while (lexer_.GetKind() != TokKind::kRbrace);
  return ParseToken(TokKind::kRbrace,
                    "expects '}' at the end of instruction list.");
}

// instruction ::= name '=' shape opcode operands
bool HloParser::ParseInstruction(HloComputation::Builder* builder) {
  string name;
  Shape shape;
  HloOpcode opcode;
  std::vector<HloInstruction*> operands;
  if (!ParseName(&name) ||
      !ParseToken(TokKind::kEqual, "expects '=' in instruction") ||
      !ParseShape(&shape) || !ParseOpcode(&opcode)) {
    return false;
  }
  switch (opcode) {
    case HloOpcode::kParameter: {
      int64 parameter_number;
      return ParseToken(TokKind::kLparen,
                        "expects '(' before parameter number") &&
             ParseInt64(&parameter_number) &&
             ParseToken(TokKind::kRparen,
                        "expects ')' after parameter number") &&
             AddInstruction(
                 name, builder->AddInstruction(HloInstruction::CreateParameter(
                           parameter_number, shape, name)));
    }
    case HloOpcode::kConstant: {
      std::unique_ptr<Literal> literal;
      return ParseToken(TokKind::kLparen,
                        "expects '(' before parameter number") &&
             ParseLiteral(&literal, shape) &&
             ParseToken(TokKind::kRparen,
                        "expects ')' after parameter number") &&
             AddInstruction(
                 name, builder->AddInstruction(
                           HloInstruction::CreateConstant(std::move(literal))));
    }
    // Unary ops.
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kBitcast:
    case HloOpcode::kCeil:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kIsFinite:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kSort:
    case HloOpcode::kTanh: {
      return ParseOperands(&operands, /*expected_size=*/1) &&
             AddInstruction(name,
                            builder->AddInstruction(HloInstruction::CreateUnary(
                                shape, opcode, operands[0])));
    }
    // Binary ops.
    case HloOpcode::kAdd:
    case HloOpcode::kDivide:
    case HloOpcode::kMultiply:
    case HloOpcode::kSubtract:
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
      return ParseOperands(&operands, /*expected_size=*/2) &&
             AddInstruction(
                 name, builder->AddInstruction(HloInstruction::CreateBinary(
                           shape, opcode, operands[0], operands[1])));
    }
    // Ternary ops.
    case HloOpcode::kClamp:
    case HloOpcode::kSelect: {
      return ParseOperands(&operands, /*expected_size=*/3) &&
             AddInstruction(
                 name,
                 builder->AddInstruction(HloInstruction::CreateTernary(
                     shape, opcode, operands[0], operands[1], operands[2])));
    }
    // Other supported ops.
    case HloOpcode::kConvert: {
      return ParseOperands(&operands, /*expected_size=*/1) &&
             AddInstruction(
                 name, builder->AddInstruction(
                           HloInstruction::CreateConvert(shape, operands[0])));
    }
    case HloOpcode::kCrossReplicaSum: {
      return ParseOperands(&operands, /*expected_size=*/1) &&
             AddInstruction(name, builder->AddInstruction(
                                      HloInstruction::CreateCrossReplicaSum(
                                          shape, operands[0])));
    }
    case HloOpcode::kReshape: {
      return ParseOperands(&operands, /*expected_size=*/1) &&
             AddInstruction(
                 name, builder->AddInstruction(
                           HloInstruction::CreateReshape(shape, operands[0])));
    }
    case HloOpcode::kBroadcast:
    case HloOpcode::kCall:
    case HloOpcode::kCustomCall:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kConvolution:
    case HloOpcode::kGetTupleElement:
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
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
    case HloOpcode::kFusion:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kRecv:
    case HloOpcode::kSend:
    case HloOpcode::kUpdate:
    case HloOpcode::kIndex:
    case HloOpcode::kTrace:
      return TokenError(StrCat("parsing not yet implemented for op: ",
                               HloOpcodeString(opcode)));
  }
}

bool HloParser::ParseLiteral(std::unique_ptr<Literal>* literal,
                             const Shape& shape) {
  switch (shape.element_type()) {
    case PRED:
      bool b;
      if (!ParseBool(&b)) {
        return false;
      }
      *literal = Literal::CreateR0<bool>(b);
      return true;
    case S32:
      int64 i;
      if (!ParseInt64(&i)) {
        return false;
      }
      *literal = Literal::CreateR0<int32>(i);
      return true;
    case F32:
      double d;
      if (!ParseDecimal(&d)) {
        return false;
      }
      *literal = Literal::CreateR0<float>(d);
      return true;
    default:
      return TokenError(StrCat("unsupported constant in shape: ",
                               ShapeUtil::HumanString(shape)));
  }
}

// operands ::= '(' operands1 ')'
// operands1
//   ::= /*empty*/
//   ::= operand (, operand)*
// operand ::= shape name
bool HloParser::ParseOperands(std::vector<HloInstruction*>* operands,
                              const int expected_size) {
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
  if (expected_size != operands->size()) {
    return TokenError(StrCat("expects ", expected_size, " operands, but has ",
                             operands->size(), " operands"));
  }
  return ParseToken(TokKind::kRparen, "expects ')' at the end of operands");
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

bool HloParser::ParseDecimal(double* result) {
  switch (lexer_.GetKind()) {
    case TokKind::kDecimal:
      *result = lexer_.GetDecimalVal();
      break;
    case TokKind::kInt:
      *result = static_cast<double>(lexer_.GetInt64Val());
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

}  // namespace

StatusOr<std::unique_ptr<HloModule>> Parse(StringPiece str) {
  HloParser parser(str);
  if (!parser.Run()) {
    return InvalidArgument("Syntax error: %s", parser.GetError().c_str());
  }
  return parser.ConsumeHloModule();
}

}  // namespace tools
}  // namespace xla
