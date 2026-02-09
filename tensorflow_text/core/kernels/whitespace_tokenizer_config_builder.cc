// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow_text/core/kernels/whitespace_tokenizer_config_builder.h"

#include <string>

#include "icu4c/source/common/unicode/uchar.h"
#include "icu4c/source/common/unicode/umachine.h"
#include "icu4c/source/common/unicode/uniset.h"
#include "icu4c/source/common/unicode/uset.h"
#include "icu4c/source/common/unicode/utf8.h"
#include "icu4c/source/common/unicode/utypes.h"

namespace tensorflow {
namespace text {

namespace {

const icu::UnicodeSet& WhiteSpaceSet() {
  // Will not fail because the data is hardcoded in the ICU library.
  UErrorCode error_code = U_ZERO_ERROR;
  const USet* c_set = u_getBinaryPropertySet(UCHAR_WHITE_SPACE, &error_code);
  // assert(U_SUCCESS(error_code));
  const icu::UnicodeSet* set = icu::UnicodeSet::fromUSet(c_set);
  return *set;
}

}  // namespace

std::string BuildWhitespaceString() {
  std::string str;
  char buf[U8_MAX_LENGTH];
  for (auto cp : WhiteSpaceSet().codePoints()) {
    int len = 0;
    U8_APPEND_UNSAFE(buf, len, cp);
    str.append(buf, len);
  }
  return str;
}

std::string BuildWhitespaceTokenizerConfig() {
  const icu::UnicodeSet& set = WhiteSpaceSet();
  int range_count = set.getRangeCount();
  UChar32 largest_whitespace = set.getRangeEnd(range_count - 1);
  // The string will hold our bit array
  std::string bitset((largest_whitespace >> 3) + 1, 0);
  for (auto cp : set.codePoints()) {
    int index = cp >> 3;
    bitset[index] |= 1 << (cp & 7);
  }
  return bitset;
}

}  // namespace text
}  // namespace tensorflow
