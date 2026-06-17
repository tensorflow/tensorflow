/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tsl/platform/str_util.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>  // NOLINT

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/platform/stringpiece.h"

namespace tsl {
namespace str_util {

size_t RemoveLeadingWhitespace(absl::string_view* text) {
  absl::string_view new_text = absl::StripLeadingAsciiWhitespace(*text);
  size_t count = text->size() - new_text.size();
  *text = new_text;
  return count;
}

size_t RemoveTrailingWhitespace(absl::string_view* text) {
  absl::string_view new_text = absl::StripTrailingAsciiWhitespace(*text);
  size_t count = text->size() - new_text.size();
  *text = new_text;
  return count;
}

size_t RemoveWhitespaceContext(absl::string_view* text) {
  absl::string_view new_text = absl::StripAsciiWhitespace(*text);
  size_t count = text->size() - new_text.size();
  *text = new_text;
  return count;
}

bool ConsumeLeadingDigits(absl::string_view* s, uint64_t* val) {
  uint64_t v;
  auto [p, ec] =
      std::from_chars(s->data(), s->data() + s->size(), v, /*base=*/10);
  if (ec != std::errc{}) {
    return false;
  }
  // Consume some digits
  s->remove_prefix(p - s->data());
  *val = v;
  return true;
}

bool ConsumeNonWhitespace(absl::string_view* s, absl::string_view* val) {
  const char* p = s->data();
  const char* limit = p + s->size();
  while (p < limit) {
    const char c = *p;
    if (absl::ascii_isspace(c)) break;
    p++;
  }
  const size_t n = p - s->data();
  if (n > 0) {
    *val = absl::string_view(s->data(), n);
    s->remove_prefix(n);
    return true;
  } else {
    *val = absl::string_view();
    return false;
  }
}

void TitlecaseString(std::string* s, absl::string_view delimiters) {
  bool upper = true;
  for (std::string::iterator ss = s->begin(); ss != s->end(); ++ss) {
    if (upper) {
      *ss = absl::ascii_toupper(*ss);
    }
    upper = absl::StrContains(delimiters, *ss);
  }
}

std::string StringReplace(absl::string_view s, absl::string_view oldsub,
                          absl::string_view newsub, bool replace_all) {
  // TODO(jlebar): We could avoid having to shift data around in the string if
  // we had a StringPiece::find() overload that searched for a StringPiece.
  std::string res(s);
  size_t pos = 0;
  while ((pos = res.find(oldsub.data(), pos, oldsub.size())) !=
         std::string::npos) {
    res.replace(pos, oldsub.size(), newsub.data(), newsub.size());
    pos += newsub.size();
    if (oldsub.empty()) {
      pos++;  // Match at the beginning of the text and after every byte
    }
    if (!replace_all) {
      break;
    }
  }
  return res;
}

std::string ArgDefCase(absl::string_view s) {
  const size_t n = s.size();

  // Compute the size of resulting string.
  // Number of extra underscores we will need to add.
  size_t extra_us = 0;
  // Number of non-alpha chars in the beginning to skip.
  size_t to_skip = 0;
  for (size_t i = 0; i < n; ++i) {
    // If we are skipping and current letter is non-alpha, skip it as well
    if (i == to_skip && !absl::ascii_isalpha(s[i])) {
      ++to_skip;
      continue;
    }

    // If we are here, we are not skipping any more.
    // If this letter is upper case, not the very first char in the
    // resulting string, and previous letter isn't replaced with an underscore,
    // we will need to insert an underscore.
    if (absl::ascii_isupper(s[i]) && i != to_skip && i > 0 &&
        absl::ascii_isalnum(s[i - 1])) {
      ++extra_us;
    }
  }

  // Initialize result with all '_'s. There is no string
  // constructor that does not initialize memory.
  std::string result(n + extra_us - to_skip, '_');
  // i - index into s
  // j - index into result
  for (size_t i = to_skip, j = 0; i < n; ++i, ++j) {
    DCHECK_LT(j, result.size());
    char c = s[i];
    // If c is not alphanumeric, we don't need to do anything
    // since there is already an underscore in its place.
    if (absl::ascii_isalnum(c)) {
      if (absl::ascii_isupper(c)) {
        // If current char is upper case, we might need to insert an
        // underscore.
        if (i != to_skip) {
          DCHECK_GT(j, 0);
          if (result[j - 1] != '_') ++j;
        }
        result[j] = absl::ascii_tolower(c);
      } else {
        result[j] = c;
      }
    }
  }

  return result;
}

}  // namespace str_util
}  // namespace tsl
