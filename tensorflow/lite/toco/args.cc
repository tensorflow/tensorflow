/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/toco/args.h"
#include "absl/strings/str_split.h"

namespace toco {
namespace {

// Helper class for SplitStructuredLine parsing.
class ClosingSymbolLookup {
 public:
  explicit ClosingSymbolLookup(const char* symbol_pairs)
      : closing_(), valid_closing_() {
    // Initialize the opening/closing arrays.
    for (const char* symbol = symbol_pairs; *symbol != 0; ++symbol) {
      unsigned char opening = *symbol;
      ++symbol;
      // If the string ends before the closing character has been found,
      // use the opening character as the closing character.
      unsigned char closing = *symbol != 0 ? *symbol : opening;
      closing_[opening] = closing;
      valid_closing_[closing] = true;
      if (*symbol == 0) break;
    }
  }

  ClosingSymbolLookup(const ClosingSymbolLookup&) = delete;
  ClosingSymbolLookup& operator=(const ClosingSymbolLookup&) = delete;

  // Returns the closing character corresponding to an opening one,
  // or 0 if the argument is not an opening character.
  char GetClosingChar(char opening) const {
    return closing_[static_cast<unsigned char>(opening)];
  }

  // Returns true if the argument is a closing character.
  bool IsClosing(char c) const {
    return valid_closing_[static_cast<unsigned char>(c)];
  }

 private:
  // Maps an opening character to its closing. If the entry contains 0,
  // the character is not in the opening set.
  char closing_[256];
  // Valid closing characters.
  bool valid_closing_[256];
};

bool SplitStructuredLine(absl::string_view line, char delimiter,
                         const char* symbol_pairs,
                         std::vector<absl::string_view>* cols) {
  ClosingSymbolLookup lookup(symbol_pairs);

  // Stack of symbols expected to close the current opened expressions.
  std::vector<char> expected_to_close;

  ABSL_RAW_CHECK(cols != nullptr, "");
  cols->push_back(line);
  for (size_t i = 0; i < line.size(); ++i) {
    char c = line[i];
    if (expected_to_close.empty() && c == delimiter) {
      // We don't have any open expression, this is a valid separator.
      cols->back().remove_suffix(line.size() - i);
      cols->push_back(line.substr(i + 1));
    } else if (!expected_to_close.empty() && c == expected_to_close.back()) {
      // Can we close the currently open expression?
      expected_to_close.pop_back();
    } else if (lookup.GetClosingChar(c)) {
      // If this is an opening symbol, we open a new expression and push
      // the expected closing symbol on the stack.
      expected_to_close.push_back(lookup.GetClosingChar(c));
    } else if (lookup.IsClosing(c)) {
      // Error: mismatched closing symbol.
      return false;
    }
  }
  if (!expected_to_close.empty()) {
    return false;  // Missing closing symbol(s)
  }
  return true;  // Success
}

inline bool TryStripPrefixString(absl::string_view str,
                                 absl::string_view prefix, string* result) {
  bool res = absl::ConsumePrefix(&str, prefix);
  result->assign(str.begin(), str.end());
  return res;
}

inline bool TryStripSuffixString(absl::string_view str,
                                 absl::string_view suffix, string* result) {
  bool res = absl::ConsumeSuffix(&str, suffix);
  result->assign(str.begin(), str.end());
  return res;
}

}  // namespace

bool Arg<toco::IntList>::Parse(string text) {
  parsed_value_.elements.clear();
  specified_ = true;
  // strings::Split("") produces {""}, but we need {} on empty input.
  // TODO(aselle): Moved this from elsewhere, but ahentz recommends we could
  // use absl::SplitLeadingDec32Values(text.c_str(), &parsed_values_.elements)
  if (!text.empty()) {
    int32 element;
    for (absl::string_view part : absl::StrSplit(text, ',')) {
      if (!SimpleAtoi(part, &element)) return false;
      parsed_value_.elements.push_back(element);
    }
  }
  return true;
}

bool Arg<toco::StringMapList>::Parse(string text) {
  parsed_value_.elements.clear();
  specified_ = true;

  if (text.empty()) {
    return true;
  }

  std::vector<absl::string_view> outer_vector;
  absl::string_view text_disposable_copy = text;
  // TODO(aselle): Change argument parsing when absl supports structuredline.
  SplitStructuredLine(text_disposable_copy, ',', "{}", &outer_vector);
  for (const absl::string_view& outer_member_stringpiece : outer_vector) {
    string outer_member(outer_member_stringpiece);
    if (outer_member.empty()) {
      continue;
    }
    string outer_member_copy = outer_member;
    absl::StripAsciiWhitespace(&outer_member);
    if (!TryStripPrefixString(outer_member, "{", &outer_member)) return false;
    if (!TryStripSuffixString(outer_member, "}", &outer_member)) return false;
    const std::vector<string> inner_fields_vector =
        absl::StrSplit(outer_member, ',');

    std::unordered_map<string, string> element;
    for (const string& member_field : inner_fields_vector) {
      std::vector<string> outer_member_key_value =
          absl::StrSplit(member_field, ':');
      if (outer_member_key_value.size() != 2) return false;
      string& key = outer_member_key_value[0];
      string& value = outer_member_key_value[1];
      absl::StripAsciiWhitespace(&key);
      absl::StripAsciiWhitespace(&value);
      if (element.count(key) != 0) return false;
      element[key] = value;
    }
    parsed_value_.elements.push_back(element);
  }
  return true;
}

}  // namespace toco
