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

#include "xla/sort_json.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace {

void SkipWhitespace(absl::string_view json, size_t& index) {
  while (index < json.size() && std::isspace(json[index])) {
    ++index;
  }
}

absl::Status CheckNotEndOfString(absl::string_view json, int index,
                                 absl::string_view expected) {
  return index < json.size()
             ? absl::OkStatus()
             : absl::InvalidArgumentError(absl::StrCat(
                   "Prematurely reached end of JSON while looking for ",
                   expected, "."));
}

absl::Status Consume(absl::string_view json, size_t& index, char c,
                     bool optional = false) {
  SkipWhitespace(json, index);
  TF_RETURN_IF_ERROR(CheckNotEndOfString(json, index, std::string(1, c)));
  if (json[index] == c) {
    ++index;
    SkipWhitespace(json, index);
  } else if (!optional) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected '", std::string(1, c), "', but found '",
                     std::string(1, json[index]), "'."));
  }
  return absl::OkStatus();
}

struct JsonArray;
struct JsonObject;

using JsonValue = std::variant<absl::string_view, std::unique_ptr<JsonObject>,
                               std::unique_ptr<JsonArray>>;

struct JsonField {
  absl::string_view name;
  JsonValue value;
};

template <typename T>
struct JsonSequence {
  std::vector<T> elements;
};

struct JsonArray : public JsonSequence<JsonValue> {};
struct JsonObject : public JsonSequence<JsonField> {};

// This parses either an array or an object.
template <typename T, char begin, char end, const char* name, typename ElemFn>
absl::StatusOr<std::unique_ptr<T>> ParseSequence(absl::string_view outer_json,
                                                 size_t& index,
                                                 ElemFn elem_fn) {
  TF_RETURN_IF_ERROR(Consume(outer_json, index, begin));
  TF_RETURN_IF_ERROR(CheckNotEndOfString(outer_json, index, name));

  auto seq = std::make_unique<T>();
  while (outer_json[index] != end) {
    TF_ASSIGN_OR_RETURN(auto elem, elem_fn(outer_json, index));
    seq->elements.emplace_back(std::move(elem));
    TF_RETURN_IF_ERROR(Consume(outer_json, index, ',', /*optional=*/true));
    TF_RETURN_IF_ERROR(CheckNotEndOfString(outer_json, index, name));
  }
  TF_RETURN_IF_ERROR(Consume(outer_json, index, end));
  return seq;
}

absl::Status EnsureValidLiteralStart(char c) {
  if (c != '"' && c != '+' && c != '-' && c != 'f' && c != 't' && c != 'n' &&
      (c < '0' || c > '9')) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid first character of literal: '", std::string(1, c), "'."));
  }
  return absl::OkStatus();
}

bool HandleEscape(absl::string_view outer_json, size_t& index,
                  bool& is_escaped) {
  if (is_escaped) {
    is_escaped = false;
    ++index;
    return true;
  }

  if (outer_json[index] == '\\') {
    is_escaped = true;
    ++index;
    return true;
  }
  return false;
}

bool LiteralIsFinished(absl::string_view outer_json, size_t& index,
                       bool is_string_literal) {
  char c = outer_json[index];
  if (is_string_literal) {
    index += (c == '"' ? 1 : 0);
    return c == '"';
  }

  return std::isspace(c) || c == ',' || c == '{' || c == '}' || c == '[' ||
         c == ']' || c == ':';
}

absl::StatusOr<absl::string_view> ParseLiteral(absl::string_view outer_json,
                                               size_t& index) {
  SkipWhitespace(outer_json, index);
  TF_RETURN_IF_ERROR(CheckNotEndOfString(outer_json, index, "literal"));

  auto c = outer_json[index];
  TF_RETURN_IF_ERROR(EnsureValidLiteralStart(c));
  bool is_string_literal = c == '"';
  size_t start_index = index;
  bool is_escaped = false;
  ++index;

  while (index < outer_json.size()) {
    if (HandleEscape(outer_json, index, is_escaped)) {
      continue;
    }
    if (LiteralIsFinished(outer_json, index, is_string_literal)) {
      break;
    }
    ++index;
  }
  return outer_json.substr(start_index, index - start_index);
}

absl::StatusOr<JsonField> ParseField(absl::string_view outer_json,
                                     size_t& index);

absl::StatusOr<JsonValue> ParseValue(absl::string_view outer_json,
                                     size_t& index) {
  JsonValue value;
  SkipWhitespace(outer_json, index);
  TF_RETURN_IF_ERROR(CheckNotEndOfString(outer_json, index, "value"));
  auto c = outer_json[index];
  if (c == '{') {
    constexpr static char kObject[] = "object";
    auto seq = ParseSequence<JsonObject, '{', '}', kObject>(outer_json, index,
                                                            ParseField);
    TF_ASSIGN_OR_RETURN(value, std::move(seq));
  } else if (c == '[') {
    constexpr static char kArray[] = "array";
    auto seq = ParseSequence<JsonArray, '[', ']', kArray>(outer_json, index,
                                                          ParseValue);
    TF_ASSIGN_OR_RETURN(value, std::move(seq));
  } else {
    TF_ASSIGN_OR_RETURN(value, ParseLiteral(outer_json, index));
  }
  return value;
}

absl::StatusOr<JsonField> ParseField(absl::string_view outer_json,
                                     size_t& index) {
  JsonField field;
  TF_ASSIGN_OR_RETURN(field.name, ParseLiteral(outer_json, index));
  TF_RETURN_IF_ERROR(Consume(outer_json, index, ':'));
  TF_ASSIGN_OR_RETURN(field.value, ParseValue(outer_json, index));
  return field;
}

template <typename T>
std::vector<std::string> SerializedElements(const JsonSequence<T>& seq) {
  std::vector<std::string> result;
  for (const auto& field : seq.elements) {
    result.push_back("");
    Serialize(field, result.back());
  }
  return result;
}

template <typename ElemT, char begin_brace, char end_brace>
void Serialize(const JsonSequence<ElemT>& object, std::string& result) {
  auto elems = SerializedElements(object);
  if constexpr (std::is_same_v<ElemT, JsonField>) {
    std::sort(elems.begin(), elems.end());
  }

  result += begin_brace;
  bool has_preceeding = false;
  for (const auto& elem : elems) {
    if (has_preceeding) {
      result += ',';
    }
    result += elem;
    has_preceeding = true;
  }
  result += end_brace;
}

void Serialize(const JsonValue& value, std::string& result) {
  if (auto* lit = std::get_if<absl::string_view>(&value)) {
    absl::StrAppend(&result, *lit);
  } else if (auto* object = std::get_if<std::unique_ptr<JsonObject>>(&value)) {
    Serialize<JsonField, '{', '}'>(**object, result);
  } else if (auto* array = std::get_if<std::unique_ptr<JsonArray>>(&value)) {
    Serialize<JsonValue, '[', ']'>(**array, result);
  }
}

void Serialize(const JsonField& field, std::string& result) {
  absl::StrAppend(&result, field.name, ":");
  Serialize(field.value, result);
}

}  // namespace

namespace xla {
absl::StatusOr<std::string> SortJson(absl::string_view json) {
  size_t index = 0;
  TF_ASSIGN_OR_RETURN(auto value, ParseValue(json, index));
  SkipWhitespace(json, index);
  if (index < json.size()) {
    return absl::InvalidArgumentError("Found trailing characters in JSON.");
  }
  std::string result;
  Serialize(value, result);
  return result;
}
}  // namespace xla
