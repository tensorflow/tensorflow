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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_PARSER_HLO_LEXER_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_PARSER_HLO_LEXER_H_

#include <string>

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/tools/parser/hlo_token.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace tools {

// Lexer for the HloModule::ToString() format text.
class HloLexer {
 public:
  explicit HloLexer(tensorflow::StringPiece buf) : buf_(buf) {
    current_ptr_ = buf_.begin();
  }

  TokKind Lex() { return current_kind_ = LexToken(); }

  TokKind GetKind() const { return current_kind_; }
  string GetStrVal() const {
    switch (GetKind()) {
      case TokKind::kName:
      case TokKind::kAttributeName:
      case TokKind::kDimLabels:
      case TokKind::kDxD:
      case TokKind::kPad:
      case TokKind::kString:
        return str_val_;
      default:
        LOG(FATAL) << "This token does not have string value";
    }
  }
  Shape GetShapeVal() const {
    CHECK(GetKind() == TokKind::kShape);
    return shape_val_;
  }
  HloOpcode GetOpcodeVal() const {
    CHECK(GetKind() == TokKind::kOpcode);
    return opcode_val_;
  }
  HloInstruction::FusionKind GetFusionKindVal() const {
    CHECK(GetKind() == TokKind::kFusionKind);
    return fusion_kind_val_;
  }
  int64 GetInt64Val() const {
    CHECK(GetKind() == TokKind::kInt);
    return int64_val_;
  }
  double GetDecimalVal() const {
    CHECK(GetKind() == TokKind::kDecimal);
    return decimal_val_;
  }

  // Returns the line of text that is currently being lexed.
  tensorflow::StringPiece GetCurrentLine() const;

 private:
  // Returns the current character. If it's neither the end of input buffer nor
  // an invalid character, moves the pointer forward.
  int GetNextChar();

  // Returns the current character.
  int PeekCurrentChar() const;

  // Creates StringPiece with the given begin and end. Exits if the begin > end,
  // or it's out of the range of the current buffer.
  tensorflow::StringPiece StringPieceFromPointers(const char* begin,
                                                  const char* end) const;
  tensorflow::RegexpStringPiece RegexpStringPieceFromPointers(
      const char* begin, const char* end) const;

  // Returns true if the given ptr is dereferenceable within the range of the
  // current buffer.
  bool CanDereference(const char* ptr) const;

  TokKind LexToken();

  TokKind LexIdentifier();
  TokKind LexPercent();
  TokKind LexShape();
  TokKind LexConstant();
  TokKind LexNumberOrPattern();
  TokKind LexComment();
  TokKind LexString();

  const tensorflow::StringPiece buf_;
  const char* current_ptr_;

  // Information about the current token.
  const char* token_start_;
  TokKind current_kind_;
  string str_val_;
  Shape shape_val_;
  HloOpcode opcode_val_;
  HloInstruction::FusionKind fusion_kind_val_;
  int64 int64_val_;
  double decimal_val_;
};

}  // namespace tools
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_PARSER_HLO_LEXER_H_
