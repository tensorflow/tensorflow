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

#include <cctype>
#include <cstdint>
#include <string>

#include "absl/strings/ascii.h"
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
  const char* p = s->data();
  const char* limit = p + s->size();
  uint64_t v = 0;
  while (p < limit) {
    const char c = *p;
    if (c < '0' || c > '9') break;
    uint64_t new_v = (v * 10) + (c - '0');
    if (new_v / 8 < v) {
      // Overflow occurred
      return false;
    }
    v = new_v;
    p++;
  }
  if (p > s->data()) {
    // Consume some digits
    s->remove_prefix(p - s->data());
    *val = v;
    return true;
  } else {
    return false;
  }
}

bool ConsumeNonWhitespace(absl::string_view* s, absl::string_view* val) {
  const char* p = s->data();
  const char* limit = p + s->size();
  while (p < limit) {
    const char c = *p;
    if (isspace(c)) break;
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

void TitlecaseString(string* s, absl::string_view delimiters) {
  bool upper = true;
  for (string::iterator ss = s->begin(); ss != s->end(); ++ss) {
    if (upper) {
      *ss = toupper(*ss);
    }
    upper = (delimiters.find(*ss) != absl::string_view::npos);
  }
}

string StringReplace(absl::string_view s, absl::string_view oldsub,
                     absl::string_view newsub, bool replace_all) {
  // TODO(jlebar): We could avoid having to shift data around in the string if
  // we had a StringPiece::find() overload that searched for a StringPiece.
  string res(s);
  size_t pos = 0;
  while ((pos = res.find(oldsub.data(), pos, oldsub.size())) != string::npos) {
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

size_t Strnlen(const char* str, const size_t string_max_len) {
  size_t len = 0;
  while (len < string_max_len && str[len] != '\0') {
    ++len;
  }
  return len;
}

string ArgDefCase(absl::string_view s) {
  const size_t n = s.size();

  // Compute the size of resulting string.
  // Number of extra underscores we will need to add.
  size_t extra_us = 0;
  // Number of non-alpha chars in the beginning to skip.
  size_t to_skip = 0;
  for (size_t i = 0; i < n; ++i) {
    // If we are skipping and current letter is non-alpha, skip it as well
    if (i == to_skip && !isalpha(s[i])) {
      ++to_skip;
      continue;
    }

    // If we are here, we are not skipping any more.
    // If this letter is upper case, not the very first char in the
    // resulting string, and previous letter isn't replaced with an underscore,
    // we will need to insert an underscore.
    if (isupper(s[i]) && i != to_skip && i > 0 && isalnum(s[i - 1])) {
      ++extra_us;
    }
  }

  // Initialize result with all '_'s. There is no string
  // constructor that does not initialize memory.
  string result(n + extra_us - to_skip, '_');
  // i - index into s
  // j - index into result
  for (size_t i = to_skip, j = 0; i < n; ++i, ++j) {
    DCHECK_LT(j, result.size());
    char c = s[i];
    // If c is not alphanumeric, we don't need to do anything
    // since there is already an underscore in its place.
    if (isalnum(c)) {
      if (isupper(c)) {
        // If current char is upper case, we might need to insert an
        // underscore.
        if (i != to_skip) {
          DCHECK_GT(j, 0);
          if (result[j - 1] != '_') ++j;
        }
        result[j] = tolower(c);
      } else {
        result[j] = c;
      }
    }
  }

  return result;
}

}  // namespace str_util
}  // namespace tsl
