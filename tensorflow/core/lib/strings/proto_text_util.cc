/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/lib/strings/proto_text_util.h"

namespace tensorflow {
namespace strings {

bool ProtoParseBoolFromScanner(Scanner* scanner, bool* value) {
  StringPiece bool_str;
  if (!scanner->RestartCapture()
           .Many(Scanner::LETTER_DIGIT)
           .GetResult(nullptr, &bool_str)) {
    return false;
  }
  ProtoSpaceAndComments(scanner);
  if (bool_str == "false" || bool_str == "False" || bool_str == "0") {
    *value = false;
    return true;
  } else if (bool_str == "true" || bool_str == "True" || bool_str == "1") {
    *value = true;
    return true;
  } else {
    return false;
  }
}

bool ProtoParseStringLiteralFromScanner(Scanner* scanner, string* value) {
  const char quote = scanner->Peek();
  if (quote != '\'' && quote != '"') return false;

  StringPiece value_sp;
  if (!scanner->One(Scanner::ALL)
           .RestartCapture()
           .ScanEscapedUntil(quote)
           .StopCapture()
           .One(Scanner::ALL)
           .GetResult(nullptr, &value_sp)) {
    return false;
  }
  ProtoSpaceAndComments(scanner);
  return str_util::CUnescape(value_sp, value, nullptr /* error */);
}

}  // namespace strings
}  // namespace tensorflow
