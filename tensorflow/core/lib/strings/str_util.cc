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

#include "tensorflow/core/lib/strings/str_util.h"

#include <ctype.h>
#include <algorithm>
#include <vector>
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace str_util {

static char hex_char[] = "0123456789abcdef";

string CEscape(StringPiece src) {
  string dest;

  for (unsigned char c : src) {
    switch (c) {
      case '\n':
        dest.append("\\n");
        break;
      case '\r':
        dest.append("\\r");
        break;
      case '\t':
        dest.append("\\t");
        break;
      case '\"':
        dest.append("\\\"");
        break;
      case '\'':
        dest.append("\\'");
        break;
      case '\\':
        dest.append("\\\\");
        break;
      default:
        // Note that if we emit \xNN and the src character after that is a hex
        // digit then that digit must be escaped too to prevent it being
        // interpreted as part of the character code by C.
        if ((c >= 0x80) || !isprint(c)) {
          dest.append("\\");
          dest.push_back(hex_char[c / 64]);
          dest.push_back(hex_char[(c % 64) / 8]);
          dest.push_back(hex_char[c % 8]);
        } else {
          dest.push_back(c);
          break;
        }
    }
  }

  return dest;
}

namespace {  // Private helpers for CUnescape().

inline bool is_octal_digit(unsigned char c) { return c >= '0' && c <= '7'; }

inline bool ascii_isxdigit(unsigned char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
         (c >= 'A' && c <= 'F');
}

inline int hex_digit_to_int(char c) {
  int x = static_cast<unsigned char>(c);
  if (x > '9') {
    x += 9;
  }
  return x & 0xf;
}

bool CUnescapeInternal(StringPiece source, string* dest,
                       string::size_type* dest_len, string* error) {
  const char* p = source.data();
  const char* end = source.end();
  const char* last_byte = end - 1;

  // We are going to write the result to dest with its iterator. If our string
  // implementation uses copy-on-write, this will trigger a copy-on-write of
  // dest's buffer; that is, dest will be assigned a new buffer.
  //
  // Note that the following way is NOT a legal way to modify a string's
  // content:
  //
  //  char* d = const_cast<char*>(dest->data());
  //
  // This won't trigger copy-on-write of the string, and so is dangerous when
  // the buffer is shared.
  auto d = dest->begin();

  // Small optimization for case where source = dest and there's no escaping
  if (source.data() == dest->data()) {
    while (p < end && *p != '\\') {
      p++;
      d++;
    }
  }

  while (p < end) {
    if (*p != '\\') {
      *d++ = *p++;
    } else {
      if (++p > last_byte) {  // skip past the '\\'
        if (error) *error = "String cannot end with \\";
        return false;
      }
      switch (*p) {
        case 'a':
          *d++ = '\a';
          break;
        case 'b':
          *d++ = '\b';
          break;
        case 'f':
          *d++ = '\f';
          break;
        case 'n':
          *d++ = '\n';
          break;
        case 'r':
          *d++ = '\r';
          break;
        case 't':
          *d++ = '\t';
          break;
        case 'v':
          *d++ = '\v';
          break;
        case '\\':
          *d++ = '\\';
          break;
        case '?':
          *d++ = '\?';
          break;  // \?  Who knew?
        case '\'':
          *d++ = '\'';
          break;
        case '"':
          *d++ = '\"';
          break;
        case '0':
        case '1':
        case '2':
        case '3':  // octal digit: 1 to 3 digits
        case '4':
        case '5':
        case '6':
        case '7': {
          const char* octal_start = p;
          unsigned int ch = *p - '0';
          if (p < last_byte && is_octal_digit(p[1])) ch = ch * 8 + *++p - '0';
          if (p < last_byte && is_octal_digit(p[1]))
            ch = ch * 8 + *++p - '0';  // now points at last digit
          if (ch > 0xff) {
            if (error) {
              *error = "Value of \\" +
                       string(octal_start, p + 1 - octal_start) +
                       " exceeds 0xff";
            }
            return false;
          }
          *d++ = ch;
          break;
        }
        case 'x':
        case 'X': {
          if (p >= last_byte) {
            if (error) *error = "String cannot end with \\x";
            return false;
          } else if (!ascii_isxdigit(p[1])) {
            if (error) *error = "\\x cannot be followed by a non-hex digit";
            return false;
          }
          unsigned int ch = 0;
          const char* hex_start = p;
          while (p < last_byte && ascii_isxdigit(p[1]))
            // Arbitrarily many hex digits
            ch = (ch << 4) + hex_digit_to_int(*++p);
          if (ch > 0xFF) {
            if (error) {
              *error = "Value of \\" + string(hex_start, p + 1 - hex_start) +
                       " exceeds 0xff";
            }
            return false;
          }
          *d++ = ch;
          break;
        }
        default: {
          if (error) *error = string("Unknown escape sequence: \\") + *p;
          return false;
        }
      }
      p++;  // read past letter we escaped
    }
  }
  *dest_len = d - dest->begin();
  return true;
}

template <typename T>
bool SplitAndParseAsInts(StringPiece text, char delim,
                         std::function<bool(StringPiece, T*)> converter,
                         std::vector<T>* result) {
  result->clear();
  std::vector<string> num_strings = Split(text, delim);
  for (const auto& s : num_strings) {
    T num;
    if (!converter(s, &num)) return false;
    result->push_back(num);
  }
  return true;
}

}  // namespace

bool CUnescape(StringPiece source, string* dest, string* error) {
  dest->resize(source.size());
  string::size_type dest_size;
  if (!CUnescapeInternal(source, dest, &dest_size, error)) {
    return false;
  }
  dest->erase(dest_size);
  return true;
}

void StripTrailingWhitespace(string* s) {
  string::size_type i;
  for (i = s->size(); i > 0 && isspace((*s)[i - 1]); --i) {
  }
  s->resize(i);
}

// Return lower-cased version of s.
string Lowercase(StringPiece s) {
  string result(s.data(), s.size());
  for (char& c : result) {
    c = tolower(c);
  }
  return result;
}

// Return upper-cased version of s.
string Uppercase(StringPiece s) {
  string result(s.data(), s.size());
  for (char& c : result) {
    c = toupper(c);
  }
  return result;
}

string ArgDefCase(StringPiece s) {
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

void TitlecaseString(string* s, StringPiece delimiters) {
  bool upper = true;
  for (string::iterator ss = s->begin(); ss != s->end(); ++ss) {
    if (upper) {
      *ss = toupper(*ss);
    }
    upper = (delimiters.find(*ss) != StringPiece::npos);
  }
}

string StringReplace(StringPiece s, StringPiece oldsub, StringPiece newsub,
                     bool replace_all) {
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

size_t RemoveLeadingWhitespace(StringPiece* text) {
  size_t count = 0;
  const char* ptr = text->data();
  while (count < text->size() && isspace(*ptr)) {
    count++;
    ptr++;
  }
  text->remove_prefix(count);
  return count;
}

size_t RemoveTrailingWhitespace(StringPiece* text) {
  size_t count = 0;
  const char* ptr = text->data() + text->size() - 1;
  while (count < text->size() && isspace(*ptr)) {
    ++count;
    --ptr;
  }
  text->remove_suffix(count);
  return count;
}

size_t RemoveWhitespaceContext(StringPiece* text) {
  // use RemoveLeadingWhitespace() and RemoveTrailingWhitespace() to do the job
  return (RemoveLeadingWhitespace(text) + RemoveTrailingWhitespace(text));
}

bool ConsumePrefix(StringPiece* s, StringPiece expected) {
  if (StartsWith(*s, expected)) {
    s->remove_prefix(expected.size());
    return true;
  }
  return false;
}

bool ConsumeSuffix(StringPiece* s, StringPiece expected) {
  if (EndsWith(*s, expected)) {
    s->remove_suffix(expected.size());
    return true;
  }
  return false;
}

bool ConsumeLeadingDigits(StringPiece* s, uint64* val) {
  const char* p = s->data();
  const char* limit = p + s->size();
  uint64 v = 0;
  while (p < limit) {
    const char c = *p;
    if (c < '0' || c > '9') break;
    uint64 new_v = (v * 10) + (c - '0');
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

bool ConsumeNonWhitespace(StringPiece* s, StringPiece* val) {
  const char* p = s->data();
  const char* limit = p + s->size();
  while (p < limit) {
    const char c = *p;
    if (isspace(c)) break;
    p++;
  }
  const size_t n = p - s->data();
  if (n > 0) {
    *val = StringPiece(s->data(), n);
    s->remove_prefix(n);
    return true;
  } else {
    *val = StringPiece();
    return false;
  }
}

bool SplitAndParseAsInts(StringPiece text, char delim,
                         std::vector<int32>* result) {
  return SplitAndParseAsInts<int32>(text, delim, strings::safe_strto32, result);
}

bool SplitAndParseAsInts(StringPiece text, char delim,
                         std::vector<int64>* result) {
  return SplitAndParseAsInts<int64>(text, delim, strings::safe_strto64, result);
}

bool SplitAndParseAsFloats(StringPiece text, char delim,
                           std::vector<float>* result) {
  return SplitAndParseAsInts<float>(text, delim,
                                    [](StringPiece str, float* value) {
                                      return strings::safe_strtof(str, value);
                                    },
                                    result);
}

size_t Strnlen(const char* str, const size_t string_max_len) {
  size_t len = 0;
  while (len < string_max_len && str[len] != '\0') {
    ++len;
  }
  return len;
}

bool StrContains(StringPiece haystack, StringPiece needle) {
  return std::search(haystack.begin(), haystack.end(), needle.begin(),
                     needle.end()) != haystack.end();
}

bool StartsWith(StringPiece text, StringPiece prefix) {
  return prefix.empty() ||
         (text.size() >= prefix.size() &&
          memcmp(text.data(), prefix.data(), prefix.size()) == 0);
}

bool EndsWith(StringPiece text, StringPiece suffix) {
  return suffix.empty() || (text.size() >= suffix.size() &&
                            memcmp(text.data() + (text.size() - suffix.size()),
                                   suffix.data(), suffix.size()) == 0);
}

}  // namespace str_util
}  // namespace tensorflow
