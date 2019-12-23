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
#include "tensorflow/lite/testing/tokenize.h"

#include <istream>
#include <string>

#include "tensorflow/lite/string_type.h"

namespace tflite {
namespace testing {

void Tokenize(std::istream* input, TokenProcessor* processor) {
  enum State { kBuildQuotedToken, kBuildToken, kIdle };

  std::string current_token;
  State state = kIdle;
  auto start_token = [&](char c) {
    state = kBuildToken;
    current_token.clear();
    current_token = c;
  };
  auto issue_token = [&]() {
    state = kIdle;
    processor->ConsumeToken(&current_token);
    current_token.clear();
  };
  auto start_quoted_token = [&]() {
    state = kBuildQuotedToken;
    current_token.clear();
  };
  auto issue_quoted_token = [&]() {
    state = kIdle;
    processor->ConsumeToken(&current_token);
    current_token.clear();
  };
  auto issue_delim = [&](char d) {
    current_token = string(1, d);
    processor->ConsumeToken(&current_token);
    current_token.clear();
  };
  auto is_delim = [](char c) { return c == '{' || c == '}' || c == ':'; };
  auto is_quote = [](char c) { return c == '"'; };

  for (auto it = std::istreambuf_iterator<char>(*input);
       it != std::istreambuf_iterator<char>(); ++it) {
    switch (state) {
      case kIdle:
        if (is_delim(*it)) {
          issue_delim(*it);
        } else if (is_quote(*it)) {
          start_quoted_token();
        } else if (!isspace(*it)) {
          start_token(*it);
        }
        break;
      case kBuildToken:
        if (is_delim(*it)) {
          issue_token();
          issue_delim(*it);
        } else if (is_quote(*it)) {
          issue_token();
          start_quoted_token();
        } else if (isspace(*it)) {
          issue_token();
        } else {
          current_token += *it;
        }
        break;
      case kBuildQuotedToken:
        if (is_quote(*it)) {
          issue_quoted_token();
        } else {
          current_token += *it;
        }
        break;
    }
  }
  if (state != kIdle) {
    issue_token();
  }
}

}  // namespace testing
}  // namespace tflite
