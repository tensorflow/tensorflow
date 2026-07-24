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

#include "tensorflow/core/kernels/text/whitespace_tokenizer.h"

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "unicode/appendable.h"  // from @icu
#include "unicode/bytestream.h"  // from @icu
#include "unicode/edits.h"  // from @icu
#include "unicode/normalizer2.h"  // from @icu
#include "unicode/schriter.h"  // from @icu
#include "unicode/stringoptions.h"  // from @icu
#include "unicode/stringpiece.h"  // from @icu
#include "unicode/uchar.h"  // from @icu
#include "unicode/ucnv.h"  // from @icu
#include "unicode/ucnv_err.h"  // from @icu
#include "unicode/umachine.h"  // from @icu
#include "unicode/uniset.h"  // from @icu
#include "unicode/unistr.h"  // from @icu
#include "unicode/uset.h"  // from @icu
#include "unicode/utf.h"  // from @icu
#include "unicode/utf8.h"  // from @icu
#include "unicode/utypes.h"  // from @icu

namespace tensorflow {
namespace text {

void WhitespaceTokenizer::Tokenize(const absl::string_view input,
                                   std::vector<std::string>* tokens) {
  std::vector<int> start_offsets, end_offsets;
  Tokenize(input, tokens, &start_offsets, &end_offsets);
}

void WhitespaceTokenizer::Tokenize(const absl::string_view input,
                                   std::vector<std::string>* tokens,
                                   std::vector<int>* start_offsets,
                                   std::vector<int>* end_offsets) {
  const int input_size = input.size();
  int position = 0, prev_position = 0;
  UChar32 codepoint;
  bool inside_token = false;
  while (position < input_size) {
    prev_position = position;
    U8_NEXT(input, position, input_size, codepoint);
    if (config_.IsWhitespace(codepoint)) {
      if (inside_token) {
        int end_pos = position - 1;
        end_offsets->push_back(end_pos);
        int start_pos = start_offsets->back();
        std::string token(input.substr(start_pos, end_pos - start_pos));
        tokens->push_back(token);
        inside_token = false;
      }
    } else {
      if (!inside_token) {
        start_offsets->push_back(prev_position);
        inside_token = true;
      }
    }
  }
  // save final word
  if (inside_token) {
    int end_pos = position;
    end_offsets->push_back(end_pos);
    int start_pos = start_offsets->back();
    std::string token(input.substr(start_pos, end_pos - start_pos));
    tokens->push_back(token);
  }
}

}  // namespace text
}  // namespace tensorflow
