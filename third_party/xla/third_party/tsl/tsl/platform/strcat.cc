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

#include "tsl/platform/strcat.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>

#include "absl/meta/type_traits.h"
#include "tsl/platform/logging.h"

namespace tsl {
namespace strings {

AlphaNum::AlphaNum(Hex hex) {
  char *const end = &digits_[kFastToBufferSize];
  char *writer = end;
  uint64 value = hex.value;
  uint64 width = hex.spec;
  // We accomplish minimum width by OR'ing in 0x10000 to the user's value,
  // where 0x10000 is the smallest hex number that is as wide as the user
  // asked for.
  uint64 mask = (static_cast<uint64>(1) << (width - 1) * 4) | value;
  static const char hexdigits[] = "0123456789abcdef";
  do {
    *--writer = hexdigits[value & 0xF];
    value >>= 4;
    mask >>= 4;
  } while (mask != 0);
  piece_ = absl::string_view(writer, end - writer);
}

// ----------------------------------------------------------------------
// StrCat()
//    This merges the given strings or integers, with no delimiter.  This
//    is designed to be the fastest possible way to construct a string out
//    of a mix of raw C strings, StringPieces, strings, and integer values.
// ----------------------------------------------------------------------

// Append is merely a version of memcpy that returns the address of the byte
// after the area just overwritten.  It comes in multiple flavors to minimize
// call overhead.
static char *Append1(char *out, const AlphaNum &x) {
  if (x.data() == nullptr) return out;

  memcpy(out, x.data(), x.size());
  return out + x.size();
}

static char *Append2(char *out, const AlphaNum &x1, const AlphaNum &x2) {
  if (x1.data() != nullptr) {
    memcpy(out, x1.data(), x1.size());
    out += x1.size();
  }

  if (x2.data() == nullptr) return out;

  memcpy(out, x2.data(), x2.size());
  return out + x2.size();
}

static char *Append4(char *out, const AlphaNum &x1, const AlphaNum &x2,
                     const AlphaNum &x3, const AlphaNum &x4) {
  if (x1.data() != nullptr) {
    memcpy(out, x1.data(), x1.size());
    out += x1.size();
  }

  if (x2.data() != nullptr) {
    memcpy(out, x2.data(), x2.size());
    out += x2.size();
  }

  if (x3.data() != nullptr) {
    memcpy(out, x3.data(), x3.size());
    out += x3.size();
  }

  if (x4.data() == nullptr) return out;

  memcpy(out, x4.data(), x4.size());
  return out + x4.size();
}

string StrCat(const AlphaNum &a) { return string(a.data(), a.size()); }

string StrCat(const AlphaNum &a, const AlphaNum &b) {
  string result(a.size() + b.size(), '\0');
  char *const begin = &*result.begin();
  char *out = Append2(begin, a, b);
  DCHECK_EQ(out, begin + result.size());
  return result;
}

string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c) {
  string result(a.size() + b.size() + c.size(), '\0');
  char *const begin = &*result.begin();
  char *out = Append2(begin, a, b);
  out = Append1(out, c);
  DCHECK_EQ(out, begin + result.size());
  return result;
}

string StrCat(const AlphaNum &a, const AlphaNum &b, const AlphaNum &c,
              const AlphaNum &d) {
  string result(a.size() + b.size() + c.size() + d.size(), '\0');
  char *const begin = &*result.begin();
  char *out = Append4(begin, a, b, c, d);
  DCHECK_EQ(out, begin + result.size());
  return result;
}

namespace {
// HasMember is true_type or false_type, depending on whether or not
// T has a __resize_default_init member. Resize will call the
// __resize_default_init member if it exists, and will call the resize
// member otherwise.
template <typename string_type, typename = void>
struct ResizeUninitializedTraits {
  using HasMember = std::false_type;
  static void Resize(string_type *s, size_t new_size) { s->resize(new_size); }
};

// __resize_default_init is provided by libc++ >= 8.0.
template <typename string_type>
struct ResizeUninitializedTraits<
    string_type, absl::void_t<decltype(std::declval<string_type &>()
                                           .__resize_default_init(237))> > {
  using HasMember = std::true_type;
  static void Resize(string_type *s, size_t new_size) {
    s->__resize_default_init(new_size);
  }
};

static inline void STLStringResizeUninitialized(string *s, size_t new_size) {
  ResizeUninitializedTraits<string>::Resize(s, new_size);
}

// Used to ensure exponential growth so that the amortized complexity of
// increasing the string size by a small amount is O(1), in contrast to
// O(str->size()) in the case of precise growth.
// TODO(b/217943845): Would be better to use absl::strings so we don't need to
// keep cherry-picking performance fixes.
template <typename string_type>
void STLStringReserveAmortized(string_type *s, size_t new_size) {
  const size_t cap = s->capacity();
  if (new_size > cap) {
    // Make sure to always grow by at least a factor of 2x.
    s->reserve((std::max)(new_size, 2 * cap));
  }
}

// Like STLStringResizeUninitialized(str, new_size), except guaranteed to use
// exponential growth so that the amortized complexity of increasing the string
// size by a small amount is O(1), in contrast to O(str->size()) in the case of
// precise growth.
template <typename string_type>
void STLStringResizeUninitializedAmortized(string_type *s, size_t new_size) {
  STLStringReserveAmortized(s, new_size);
  STLStringResizeUninitialized(s, new_size);
}

}  // namespace
namespace internal {

// Do not call directly - these are not part of the public API.
string CatPieces(std::initializer_list<absl::string_view> pieces) {
  size_t total_size = 0;
  for (const absl::string_view piece : pieces) total_size += piece.size();
  string result(total_size, '\0');

  char *const begin = &*result.begin();
  char *out = begin;
  for (const absl::string_view piece : pieces) {
    const size_t this_size = piece.size();
    memcpy(out, piece.data(), this_size);
    out += this_size;
  }
  DCHECK_EQ(out, begin + result.size());
  return result;
}

// It's possible to call StrAppend with a StringPiece that is itself a fragment
// of the string we're appending to.  However the results of this are random.
// Therefore, check for this in debug mode.  Use unsigned math so we only have
// to do one comparison.
#define DCHECK_NO_OVERLAP(dest, src) \
  DCHECK_GE(uintptr_t((src).data() - (dest).data()), uintptr_t((dest).size()))

void AppendPieces(string *result,
                  std::initializer_list<absl::string_view> pieces) {
  size_t old_size = result->size();
  size_t total_size = old_size;
  for (const absl::string_view piece : pieces) {
    DCHECK_NO_OVERLAP(*result, piece);
    total_size += piece.size();
  }
  STLStringResizeUninitializedAmortized(result, total_size);

  char *const begin = &*result->begin();
  char *out = begin + old_size;
  for (const absl::string_view piece : pieces) {
    const size_t this_size = piece.size();
    memcpy(out, piece.data(), this_size);
    out += this_size;
  }
  DCHECK_EQ(out, begin + result->size());
}

}  // namespace internal

void StrAppend(string *result, const AlphaNum &a) {
  DCHECK_NO_OVERLAP(*result, a);
  result->append(a.data(), a.size());
}

void StrAppend(string *result, const AlphaNum &a, const AlphaNum &b) {
  DCHECK_NO_OVERLAP(*result, a);
  DCHECK_NO_OVERLAP(*result, b);
  string::size_type old_size = result->size();
  STLStringResizeUninitializedAmortized(result, old_size + a.size() + b.size());
  char *const begin = &*result->begin();
  char *out = Append2(begin + old_size, a, b);
  DCHECK_EQ(out, begin + result->size());
}

void StrAppend(string *result, const AlphaNum &a, const AlphaNum &b,
               const AlphaNum &c) {
  DCHECK_NO_OVERLAP(*result, a);
  DCHECK_NO_OVERLAP(*result, b);
  DCHECK_NO_OVERLAP(*result, c);
  string::size_type old_size = result->size();
  STLStringResizeUninitializedAmortized(
      result, old_size + a.size() + b.size() + c.size());
  char *const begin = &*result->begin();
  char *out = Append2(begin + old_size, a, b);
  out = Append1(out, c);
  DCHECK_EQ(out, begin + result->size());
}

void StrAppend(string *result, const AlphaNum &a, const AlphaNum &b,
               const AlphaNum &c, const AlphaNum &d) {
  DCHECK_NO_OVERLAP(*result, a);
  DCHECK_NO_OVERLAP(*result, b);
  DCHECK_NO_OVERLAP(*result, c);
  DCHECK_NO_OVERLAP(*result, d);
  string::size_type old_size = result->size();
  STLStringResizeUninitializedAmortized(
      result, old_size + a.size() + b.size() + c.size() + d.size());
  char *const begin = &*result->begin();
  char *out = Append4(begin + old_size, a, b, c, d);
  DCHECK_EQ(out, begin + result->size());
}

}  // namespace strings
}  // namespace tsl
