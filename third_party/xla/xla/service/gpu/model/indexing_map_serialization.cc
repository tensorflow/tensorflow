/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/model/indexing_map_serialization.h"

#include <cctype>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {
namespace {

using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::StringRef;
using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::ArrayRef;
using mlir::MLIRContext;

enum class Delimeter { kParen, kBracket };

struct Token {
  enum class Kind {
    // Variable name, e.g. "d0", "s1".
    kVarName,
    // Integer literal.
    kIntLiteral,
    kBoolLiteral,
    // Keywords
    kKeywordDomain,
    kKeywordIn,
    kKeywordIsSimplified,
    // Arithmetic operation, e.g. "+", "-", "*", "floorDiv", "mod".
    kPlus,
    kMinus,
    kTimes,
    kFloorDiv,
    kMod,
    // Punctuation.
    kArrow,
    kLParen,
    kRParen,
    kLBracket,
    kRBracket,
    kComma,
    kColon,
    // Status.
    kError,
    kEOF
  };
  StringRef spelling;
  Token::Kind kind;
};

Token::Kind GetSingleCharTokenType(char c) {
  switch (c) {
    case '(':
      return Token::Kind::kLParen;
    case ')':
      return Token::Kind::kRParen;
    case '[':
      return Token::Kind::kLBracket;
    case ']':
      return Token::Kind::kRBracket;
    case ',':
      return Token::Kind::kComma;
    case ':':
      return Token::Kind::kColon;
    case '+':
      return Token::Kind::kPlus;
    case '-':
      return Token::Kind::kMinus;
    case '*':
      return Token::Kind::kTimes;
    default:
      return Token::Kind::kError;
  }
}

bool IsPartOfAffineExpr(Token token) {
  return token.kind == Token::Kind::kVarName ||
         token.kind == Token::Kind::kIntLiteral ||
         token.kind == Token::Kind::kPlus ||
         token.kind == Token::Kind::kMinus ||
         token.kind == Token::Kind::kTimes ||
         token.kind == Token::Kind::kFloorDiv ||
         token.kind == Token::Kind::kMod;
}

class Parser {
 public:
  explicit Parser(llvm::StringRef input) : input_(input), it_(input.begin()) {
    // Set the parser to the first token.
    Advance();
  }

  const Token& GetCurrentToken() const { return current_token_; };
  void Advance() { current_token_ = GetNextTokenImpl(); }
  Token GetNextToken() {
    Advance();
    return current_token_;
  }

  bool ConsumeToken(Token::Kind kind);
  bool ParseVarName(std::string* var_name);
  bool ParseInt(int64_t* value);
  bool ParseBool(bool* boolean);
  bool ParseInterval(Interval* interval);
  bool ParseAffineExprString(std::string* affine_expr_str);
  bool ParseCommaSeparatedVarList(
      Delimeter delimeter,
      llvm::function_ref<bool(Parser& parser)> parse_element_fn);

 private:
  void ConsumeWhitespace() {
    while (it_ != input_.end() && std::isspace(*it_)) ++it_;
  }

  // Parses the next token from the input and sets the iterator to the position
  // right after it.
  Token GetNextTokenImpl();

  llvm::StringRef input_;
  llvm::StringRef::iterator it_;
  Token current_token_;
};

bool Parser::ParseVarName(std::string* var_name) {
  if (current_token_.kind != Token::Kind::kVarName) {
    llvm::errs() << "Expected var name, got: " << current_token_.spelling
                 << "\n";
    return false;
  }
  *var_name = current_token_.spelling.str();
  Advance();
  return true;
}

bool Parser::ParseInt(int64_t* value) {
  int val;
  if (current_token_.kind != Token::Kind::kIntLiteral ||
      current_token_.spelling.getAsInteger(/*radix=*/0, val)) {
    llvm::errs() << "Expected int literal, got: " << current_token_.spelling
                 << "\n";
    return false;
  }
  *value = static_cast<int64_t>(val);
  Advance();
  return true;
}

bool Parser::ParseBool(bool* boolean) {
  if (current_token_.kind != Token::Kind::kBoolLiteral) {
    llvm::errs() << "Expected bool literal, got: " << current_token_.spelling
                 << "\n";
    return false;
  }
  *boolean = current_token_.spelling.compare("true") == 0;
  Advance();
  return true;
}

bool Parser::ParseInterval(Interval* interval) {
  if (!ConsumeToken(Token::Kind::kLBracket) || !ParseInt(&interval->lower) ||
      !ConsumeToken(Token::Kind::kComma) || !ParseInt(&interval->upper) ||
      !ConsumeToken(Token::Kind::kRBracket)) {
    return false;
  }
  return interval;
}

bool Parser::ParseAffineExprString(std::string* affine_expr_str) {
  unsigned num_unmatched_parens = 0;
  while (true) {
    if (!IsPartOfAffineExpr(current_token_)) {
      if (ConsumeToken(Token::Kind::kLParen)) {
        ++num_unmatched_parens;
      } else if (current_token_.kind == Token::Kind::kRParen &&
                 num_unmatched_parens > 0) {
        --num_unmatched_parens;
        Advance();
      } else {
        break;
      }
    }
    affine_expr_str->append(current_token_.spelling);
    affine_expr_str->push_back(' ');
    Advance();
  }
  return current_token_.kind != Token::Kind::kError;
}

bool Parser::ParseCommaSeparatedVarList(
    Delimeter delimeter,
    llvm::function_ref<bool(Parser& parser)> parse_element_fn) {
  auto left_delimiter = delimeter == Delimeter::kParen ? Token::Kind::kLParen
                                                       : Token::Kind::kLBracket;
  auto right_delimiter = delimeter == Delimeter::kParen
                             ? Token::Kind::kRParen
                             : Token::Kind::kRBracket;
  if (!ConsumeToken(left_delimiter)) {
    return false;
  }
  if (ConsumeToken(right_delimiter)) {
    return true;
  }
  std::string element;
  while (parse_element_fn(*this)) {
    if (ConsumeToken(Token::Kind::kComma)) continue;
    return ConsumeToken(right_delimiter);
  }
  return false;
}

bool Parser::ConsumeToken(Token::Kind kind) {
  Token token = GetCurrentToken();
  if (token.kind != kind) {
    return false;
  }
  GetNextToken();
  return true;
}

Token Parser::GetNextTokenImpl() {
  if (current_token_.kind == Token::Kind::kError ||
      current_token_.kind == Token::Kind::kEOF) {
    return current_token_;
  }
  ConsumeWhitespace();
  if (it_ == input_.end()) {
    return Token{"", Token::Kind::kEOF};
  }
  auto start = it_;
  if (std::isalpha(*it_)) {
    // Variable name.
    while (it_ != input_.end() &&
           (std::isalpha(*it_) || std::isdigit(*it_) || *it_ == '_')) {
      ++it_;
    }
    StringRef spelling = input_.substr(start - input_.data(), it_ - start);
    if (spelling == "true" || spelling == "false") {
      return Token{spelling, Token::Kind::kBoolLiteral};
    }
    if (spelling == "domain") {
      return Token{spelling, Token::Kind::kKeywordDomain};
    }
    if (spelling == "is_simplified") {
      return Token{spelling, Token::Kind::kKeywordIsSimplified};
    }
    if (spelling == "in") {
      return Token{spelling, Token::Kind::kKeywordIn};
    }
    if (spelling == "mod") {
      return Token{spelling, Token::Kind::kMod};
    }
    if (spelling == "floorDiv") {
      return Token{spelling, Token::Kind::kFloorDiv};
    }
    return Token{spelling, Token::Kind::kVarName};
  }
  if (std::isdigit(*it_)) {
    auto start = it_;
    while (it_ != input_.end() && std::isdigit(*it_)) {
      ++it_;
    }

    StringRef spelling = input_.substr(start - input_.data(), it_ - start);
    return Token{spelling, Token::Kind::kIntLiteral};
  }
  if (*it_ == '-') {
    ++it_;
    if (it_ != input_.end() && *it_ == '>') {
      ++it_;
      return Token{"->", Token::Kind::kArrow};
    } else {
      return Token{"-", Token::Kind::kMinus};
    }
  }
  StringRef spelling = input_.substr(start - input_.data(), 1);
  return Token{spelling, GetSingleCharTokenType(*(it_++))};
}

// Parses a comma separated list of variable names. It is used to parse the
// lists of dimension and symbol variables.
bool ParseVarNames(Parser& parser, Delimeter delimeter,
                   SmallVectorImpl<std::string>& var_names) {
  auto parse_var_name_fn = [&](Parser& parser) {
    std::string var_name;
    if (!parser.ParseVarName(&var_name)) {
      return false;
    }
    var_names.push_back(var_name);
    return true;
  };
  return parser.ParseCommaSeparatedVarList(delimeter, parse_var_name_fn);
}

// Parses a comma separated list of affine expressions. It is used to parse
// the list of affine map results.
bool ParseAffineMapResults(Parser& parser,
                           SmallVectorImpl<std::string>& affine_expr_strs) {
  auto parse_var_name_fn = [&](Parser& parser) {
    std::string affine_expr_str;
    if (!parser.ParseAffineExprString(&affine_expr_str)) {
      return false;
    }
    affine_expr_strs.push_back(affine_expr_str);
    return true;
  };
  return parser.ParseCommaSeparatedVarList(Delimeter::kParen,
                                           parse_var_name_fn);
}

// Assembles an affine map from the given dimension and symbol names and the
// affine expressions for the results.
bool ParseAffineExprsWithMLIR(ArrayRef<std::string> dim_var_names,
                              ArrayRef<std::string> symbol_var_names,
                              ArrayRef<std::string> affine_expr_strings,
                              MLIRContext* context,
                              SmallVectorImpl<AffineExpr>& affine_exprs) {
  std::stringstream ss;
  ss << "affine_map<(" << absl::StrJoin(dim_var_names, ", ") << ") ";
  if (!symbol_var_names.empty()) {
    ss << '[' << absl::StrJoin(symbol_var_names, ", ") << "] ";
  }
  ss << " -> (" << absl::StrJoin(affine_expr_strings, ", ") << ")>";
  auto affine_map_attr = mlir::parseAttribute(ss.str(), context);
  if (!affine_map_attr) {
    llvm::errs() << "Failed to parse affine map: " << ss.str() << "\n";
    return false;
  }
  mlir::AffineMap affine_map =
      mlir::cast<mlir::AffineMapAttr>(affine_map_attr).getValue();
  affine_exprs = llvm::to_vector(affine_map.getResults());
  return true;
}

}  // namespace

std::optional<IndexingMap> ParseIndexingMap(llvm::StringRef input,
                                            mlir::MLIRContext* context) {
  Parser parser(input);

  // Parse variable names.
  SmallVector<std::string, 8> dim_var_names;
  SmallVector<std::string, 4> symbol_var_names;
  if (!ParseVarNames(parser, Delimeter::kParen, dim_var_names) ||
      (parser.GetCurrentToken().kind == Token::Kind::kLBracket &&
       !ParseVarNames(parser, Delimeter::kBracket, symbol_var_names))) {
    llvm::errs() << "Failed to parse variable names\n";
    return std::nullopt;
  }

  // Parse affine map results.
  SmallVector<std::string, 3> affine_expr_strs;
  if (!parser.ConsumeToken(Token::Kind::kArrow) ||
      !ParseAffineMapResults(parser, affine_expr_strs)) {
    llvm::errs() << "Failed to parse affine map results\n";
    return std::nullopt;
  }
  int num_affine_map_results = affine_expr_strs.size();

  // Special case: no domain is printed for the empty map.
  if (dim_var_names.empty() && symbol_var_names.empty()) {
    if (num_affine_map_results != 0 ||
        parser.GetCurrentToken().kind != Token::Kind::kEOF) {
      llvm::errs() << "Expected an empty indexing map\n";
      return std::nullopt;
    }
    return IndexingMap{AffineMap::get(context), /*dimensions=*/{},
                       /*range_vars=*/{}, /*rt_vars=*/{}};
  }

  if (!parser.ConsumeToken(Token::Kind::kComma) ||
      !parser.ConsumeToken(Token::Kind::kKeywordDomain) ||
      !parser.ConsumeToken(Token::Kind::kColon)) {
    return std::nullopt;
  }
  // Parse dimension variables.
  std::vector<DimVar> dim_vars;
  for (auto& dim_name : dim_var_names) {
    std::string var_name;
    Interval interval;
    if (!parser.ParseVarName(&var_name) ||
        !parser.ConsumeToken(Token::Kind::kKeywordIn) ||
        !parser.ParseInterval(&interval) ||
        !parser.ConsumeToken(Token::Kind::kComma)) {
      return std::nullopt;
    }
    if (var_name != dim_name) {
      return std::nullopt;
    }
    dim_vars.push_back(DimVar{interval});
  }
  // Parse range variables.
  std::vector<RangeVar> range_vars;
  for (auto& symbol_var : symbol_var_names) {
    std::string var_name;
    Interval interval;
    if (!parser.ParseVarName(&var_name) ||
        !parser.ConsumeToken(Token::Kind::kKeywordIn) ||
        !parser.ParseInterval(&interval) ||
        !parser.ConsumeToken(Token::Kind::kComma)) {
      return std::nullopt;
    }
    if (var_name != symbol_var) {
      return std::nullopt;
    }
    range_vars.push_back(RangeVar{interval});
  }
  // Parse constraints.
  SmallVector<Interval> constraint_bounds;
  while (!parser.ConsumeToken(Token::Kind::kKeywordIsSimplified)) {
    std::string affine_expr_str;
    Interval interval;
    if (!parser.ParseAffineExprString(&affine_expr_str) ||
        !parser.ConsumeToken(Token::Kind::kKeywordIn) ||
        !parser.ParseInterval(&interval) ||
        !parser.ConsumeToken(Token::Kind::kComma)) {
      return std::nullopt;
    }
    affine_expr_strs.push_back(affine_expr_str);
    constraint_bounds.push_back(interval);
  }
  // Parse is_simplified.
  bool is_simplified;
  if (!parser.ConsumeToken(Token::Kind::kColon) ||
      !parser.ParseBool(&is_simplified)) {
    return std::nullopt;
  }
  // Check that the input is consumed.
  if (!parser.ConsumeToken(Token::Kind::kEOF)) {
    return std::nullopt;
  }

  // Parse affine expressions.
  SmallVector<AffineExpr> affine_exprs;
  if (!ParseAffineExprsWithMLIR(dim_var_names, symbol_var_names,
                                affine_expr_strs, context, affine_exprs)) {
    return std::nullopt;
  }
  ArrayRef<AffineExpr> affine_map_results =
      ArrayRef(affine_exprs).take_front(num_affine_map_results);
  ArrayRef<AffineExpr> constraint_exprs =
      ArrayRef(affine_exprs).drop_front(num_affine_map_results);

  // Populate constraints.
  SmallVector<std::pair<AffineExpr, Interval>> constraints;
  constraints.reserve(constraint_exprs.size());
  for (const auto& [expr, bounds] :
       llvm::zip(constraint_exprs, constraint_bounds)) {
    constraints.push_back(std::make_pair(expr, bounds));
  }
  auto map = AffineMap::get(dim_vars.size(), range_vars.size(),
                            affine_map_results, context);
  return IndexingMap{
      map,         std::move(dim_vars), std::move(range_vars), /*rt_vars=*/{},
      constraints, is_simplified};
}

}  // namespace gpu
}  // namespace xla
