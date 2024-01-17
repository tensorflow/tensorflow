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

#ifndef TENSORFLOW_TSL_PLATFORM_TSTRING_H_
#define TENSORFLOW_TSL_PLATFORM_TSTRING_H_

#include <assert.h>

#include <ostream>
#include <string>

#include "tsl/platform/cord.h"
#include "tsl/platform/ctstring.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/stringpiece.h"

namespace tsl {

// tensorflow::tstring is the scalar type for DT_STRING tensors.
//
// tstrings are meant to be used when interfacing with string tensors, and
// should not be considered as a general replacement for std::string in
// tensorflow.  The primary purpose of tstring is to provide a unified and
// stable ABI for string tensors across TF Core/C-API/Lite/etc---mitigating
// unnecessary conversions across language boundaries, and allowing for compiler
// agnostic interoperability across dynamically loaded modules.
//
// In addition to ABI stability, tstrings features two string subtypes, VIEW and
// OFFSET.
//
// VIEW tstrings are views into unowned character buffers; they can be used to
// pass around existing character strings without incurring a per object heap
// allocation.  Note that, like std::string_view, it is the user's
// responsibility to ensure that the underlying buffer of a VIEW tstring exceeds
// the lifetime of the associated tstring object.
//
// TODO(dero): Methods for creating OFFSET tensors are not currently
// implemented.
//
// OFFSET tstrings are platform independent offset defined strings which can be
// directly mmaped or copied into a tensor buffer without the need for
// deserialization or processing.  For security reasons, it is imperative that
// OFFSET based string tensors are validated before use, or are from a trusted
// source.
//
// Underlying VIEW and OFFSET buffers are considered immutable, so l-value
// assignment, mutation, or non-const access to data() of tstrings will result
// in the conversion to an owned SMALL/LARGE type.
//
// The interface for tstring largely overlaps with std::string. Except where
// noted, expect equivalent semantics with synonymous std::string methods.
class tstring {
  TF_TString tstr_;

 public:
  enum Type {
    // See cstring.h
    SMALL = TF_TSTR_SMALL,
    LARGE = TF_TSTR_LARGE,
    OFFSET = TF_TSTR_OFFSET,
    VIEW = TF_TSTR_VIEW,
  };

  // Assignment to a tstring object with a tstring::view type will create a VIEW
  // type tstring.
  class view {
    const char* data_;
    size_t size_;

   public:
    explicit view(const char* data, size_t size) : data_(data), size_(size) {}
    explicit view(const char* data) : data_(data), size_(::strlen(data)) {}

    const char* data() const { return data_; }

    size_t size() const { return size_; }

    view() = delete;
    view(const view&) = delete;
    view& operator=(const view&) = delete;
  };

  typedef const char* const_iterator;

  // Ctor
  tstring();
  tstring(const std::string& str);  // NOLINT TODO(b/147740521): Make explicit.
  tstring(const char* str, size_t len);
  tstring(const char* str);  // NOLINT TODO(b/147740521): Make explicit.
  tstring(size_t n, char c);
  explicit tstring(const StringPiece str);
#ifdef PLATFORM_GOOGLE
  explicit tstring(const absl::Cord& cord);
#endif  // PLATFORM_GOOGLE

  // Copy
  tstring(const tstring& str);

  // Move
  tstring(tstring&& str) noexcept;

  // Dtor
  ~tstring();

  // Copy Assignment
  tstring& operator=(const tstring& str);
  tstring& operator=(const std::string& str);
  tstring& operator=(const char* str);
  tstring& operator=(char ch);
  tstring& operator=(const StringPiece str);
#ifdef PLATFORM_GOOGLE
  tstring& operator=(const absl::Cord& cord);
#endif  // PLATFORM_GOOGLE

  // View Assignment
  tstring& operator=(const view& tsv);

  // Move Assignment
  tstring& operator=(tstring&& str);

  // Comparison
  int compare(const char* str, size_t len) const;
  bool operator<(const tstring& o) const;
  bool operator>(const tstring& o) const;
  bool operator==(const char* str) const;
  bool operator==(const tstring& o) const;
  bool operator!=(const char* str) const;
  bool operator!=(const tstring& o) const;

  // Conversion Operators
  // TODO(b/147740521): Make explicit.
  operator std::string() const;  // NOLINT
  // TODO(b/147740521): Make explicit.
  operator StringPiece() const;  // NOLINT
#ifdef PLATFORM_GOOGLE
  template <typename T,
            typename std::enable_if<std::is_same<T, absl::AlphaNum>::value,
                                    T>::type* = nullptr>
  operator T() const;  // NOLINT TODO(b/147740521): Remove.
#endif  // PLATFORM_GOOGLE

  // Attributes
  size_t size() const;
  size_t length() const;
  size_t capacity() const;
  bool empty() const;
  Type type() const;

  // Allocation
  void resize(size_t new_size, char c = 0);
  // Similar to resize, but will leave the newly grown region uninitialized.
  void resize_uninitialized(size_t new_size);
  void clear() noexcept;
  void reserve(size_t n);

  // Iterators
  const_iterator begin() const;
  const_iterator end() const;

  // Const Element Access
  const char* c_str() const;
  const char* data() const;
  const char& operator[](size_t i) const;
  const char& back() const;

  // Mutable Element Access
  // NOTE: For VIEW/OFFSET types, calling these methods will result in the
  // conversion to a SMALL or heap allocated LARGE type.  As a result,
  // previously obtained pointers, references, or iterators to the underlying
  // buffer will point to the original VIEW/OFFSET and not the new allocation.
  char* mdata();
  char* data();  // DEPRECATED: Use mdata().
  char& operator[](size_t i);

  // Assignment
  tstring& assign(const char* str, size_t len);
  tstring& assign(const char* str);

  // View Assignment
  tstring& assign_as_view(const tstring& str);
  tstring& assign_as_view(const std::string& str);
  tstring& assign_as_view(const StringPiece str);
  tstring& assign_as_view(const char* str, size_t len);
  tstring& assign_as_view(const char* str);

  // Modifiers
  // NOTE: Invalid input will result in undefined behavior.
  tstring& append(const tstring& str);
  tstring& append(const char* str, size_t len);
  tstring& append(const char* str);
  tstring& append(size_t n, char c);

  tstring& erase(size_t pos, size_t len);

  tstring& insert(size_t pos, const tstring& str, size_t subpos, size_t sublen);
  tstring& insert(size_t pos, size_t n, char c);
  void swap(tstring& str);
  void push_back(char ch);

  // Friends
  friend bool operator==(const char* a, const tstring& b);
  friend bool operator==(const std::string& a, const tstring& b);
  friend tstring operator+(const tstring& a, const tstring& b);
  friend std::ostream& operator<<(std::ostream& o, const tstring& str);
  friend std::hash<tstring>;
};

// Non-member function overloads

bool operator==(const char* a, const tstring& b);
bool operator==(const std::string& a, const tstring& b);
tstring operator+(const tstring& a, const tstring& b);
std::ostream& operator<<(std::ostream& o, const tstring& str);

// Implementations

// Ctor

inline tstring::tstring() { TF_TString_Init(&tstr_); }

inline tstring::tstring(const char* str, size_t len) {
  TF_TString_Init(&tstr_);
  TF_TString_Copy(&tstr_, str, len);
}

inline tstring::tstring(const char* str) : tstring(str, ::strlen(str)) {}

inline tstring::tstring(size_t n, char c) {
  TF_TString_Init(&tstr_);
  TF_TString_Resize(&tstr_, n, c);
}

inline tstring::tstring(const std::string& str)
    : tstring(str.data(), str.size()) {}

inline tstring::tstring(const StringPiece str)
    : tstring(str.data(), str.size()) {}

#ifdef PLATFORM_GOOGLE
inline tstring::tstring(const absl::Cord& cord) {
  TF_TString_Init(&tstr_);
  TF_TString_ResizeUninitialized(&tstr_, cord.size());

  cord.CopyToArray(data());
}
#endif  // PLATFORM_GOOGLE

// Copy

inline tstring::tstring(const tstring& str) {
  TF_TString_Init(&tstr_);
  TF_TString_Assign(&tstr_, &str.tstr_);
}

// Move

inline tstring::tstring(tstring&& str) noexcept {
  TF_TString_Init(&tstr_);
  TF_TString_Move(&tstr_, &str.tstr_);
}

// Dtor

inline tstring::~tstring() { TF_TString_Dealloc(&tstr_); }

// Copy Assignment

inline tstring& tstring::operator=(const tstring& str) {
  TF_TString_Assign(&tstr_, &str.tstr_);

  return *this;
}

inline tstring& tstring::operator=(const std::string& str) {
  TF_TString_Copy(&tstr_, str.data(), str.size());
  return *this;
}

inline tstring& tstring::operator=(const char* str) {
  TF_TString_Copy(&tstr_, str, ::strlen(str));

  return *this;
}

inline tstring& tstring::operator=(char c) {
  resize_uninitialized(1);
  (*this)[0] = c;

  return *this;
}

inline tstring& tstring::operator=(const StringPiece str) {
  TF_TString_Copy(&tstr_, str.data(), str.size());

  return *this;
}

#ifdef PLATFORM_GOOGLE
inline tstring& tstring::operator=(const absl::Cord& cord) {
  TF_TString_ResizeUninitialized(&tstr_, cord.size());

  cord.CopyToArray(data());

  return *this;
}
#endif  // PLATFORM_GOOGLE

// View Assignment

inline tstring& tstring::operator=(const tstring::view& tsv) {
  assign_as_view(tsv.data(), tsv.size());

  return *this;
}

// Move Assignment

inline tstring& tstring::operator=(tstring&& str) {
  TF_TString_Move(&tstr_, &str.tstr_);

  return *this;
}

// Comparison

inline int tstring::compare(const char* str, size_t len) const {
  int ret = ::memcmp(data(), str, std::min(len, size()));

  if (ret < 0) return -1;
  if (ret > 0) return +1;

  if (size() < len) return -1;
  if (size() > len) return +1;

  return 0;
}

inline bool tstring::operator<(const tstring& o) const {
  return compare(o.data(), o.size()) < 0;
}

inline bool tstring::operator>(const tstring& o) const {
  return compare(o.data(), o.size()) > 0;
}

inline bool tstring::operator==(const char* str) const {
  return ::strlen(str) == size() && ::memcmp(data(), str, size()) == 0;
}

inline bool tstring::operator==(const tstring& o) const {
  return o.size() == size() && ::memcmp(data(), o.data(), size()) == 0;
}

inline bool tstring::operator!=(const char* str) const {
  return !(*this == str);
}

inline bool tstring::operator!=(const tstring& o) const {
  return !(*this == o);
}

// Conversion Operators

inline tstring::operator std::string() const {
  return std::string(data(), size());
}

inline tstring::operator StringPiece() const {
  return StringPiece(data(), size());
}

#ifdef PLATFORM_GOOGLE
template <typename T, typename std::enable_if<
                          std::is_same<T, absl::AlphaNum>::value, T>::type*>
inline tstring::operator T() const {
  return T(StringPiece(*this));
}
#endif  // PLATFORM_GOOGLE

// Attributes

inline size_t tstring::size() const { return TF_TString_GetSize(&tstr_); }

inline size_t tstring::length() const { return size(); }

inline size_t tstring::capacity() const {
  return TF_TString_GetCapacity(&tstr_);
}

inline bool tstring::empty() const { return size() == 0; }

inline tstring::Type tstring::type() const {
  return static_cast<tstring::Type>(TF_TString_GetType(&tstr_));
}

// Allocation

inline void tstring::resize(size_t new_size, char c) {
  TF_TString_Resize(&tstr_, new_size, c);
}

inline void tstring::resize_uninitialized(size_t new_size) {
  TF_TString_ResizeUninitialized(&tstr_, new_size);
}

inline void tstring::clear() noexcept {
  TF_TString_ResizeUninitialized(&tstr_, 0);
}

inline void tstring::reserve(size_t n) { TF_TString_Reserve(&tstr_, n); }

// Iterators

inline tstring::const_iterator tstring::begin() const { return &(*this)[0]; }
inline tstring::const_iterator tstring::end() const { return &(*this)[size()]; }

// Element Access

inline const char* tstring::c_str() const { return data(); }

inline const char* tstring::data() const {
  return TF_TString_GetDataPointer(&tstr_);
}

inline const char& tstring::operator[](size_t i) const { return data()[i]; }

inline const char& tstring::back() const { return (*this)[size() - 1]; }

inline char* tstring::mdata() {
  return TF_TString_GetMutableDataPointer(&tstr_);
}

inline char* tstring::data() {
  // Deprecated
  return mdata();
}

inline char& tstring::operator[](size_t i) { return mdata()[i]; }

// Assignment

inline tstring& tstring::assign(const char* str, size_t len) {
  TF_TString_Copy(&tstr_, str, len);

  return *this;
}

inline tstring& tstring::assign(const char* str) {
  assign(str, ::strlen(str));

  return *this;
}

// View Assignment

inline tstring& tstring::assign_as_view(const tstring& str) {
  assign_as_view(str.data(), str.size());

  return *this;
}

inline tstring& tstring::assign_as_view(const std::string& str) {
  assign_as_view(str.data(), str.size());

  return *this;
}

inline tstring& tstring::assign_as_view(const StringPiece str) {
  assign_as_view(str.data(), str.size());

  return *this;
}

inline tstring& tstring::assign_as_view(const char* str, size_t len) {
  TF_TString_AssignView(&tstr_, str, len);

  return *this;
}

inline tstring& tstring::assign_as_view(const char* str) {
  assign_as_view(str, ::strlen(str));

  return *this;
}

// Modifiers

inline tstring& tstring::append(const tstring& str) {
  TF_TString_Append(&tstr_, &str.tstr_);

  return *this;
}

inline tstring& tstring::append(const char* str, size_t len) {
  TF_TString_AppendN(&tstr_, str, len);

  return *this;
}

inline tstring& tstring::append(const char* str) {
  append(str, ::strlen(str));

  return *this;
}

inline tstring& tstring::append(size_t n, char c) {
  // For append use cases, we want to ensure amortized growth.
  const size_t new_size = size() + n;
  TF_TString_ReserveAmortized(&tstr_, new_size);
  resize(new_size, c);

  return *this;
}

inline tstring& tstring::erase(size_t pos, size_t len) {
  memmove(mdata() + pos, data() + pos + len, size() - len - pos);

  resize(size() - len);

  return *this;
}

inline tstring& tstring::insert(size_t pos, const tstring& str, size_t subpos,
                                size_t sublen) {
  size_t orig_size = size();
  TF_TString_ResizeUninitialized(&tstr_, orig_size + sublen);

  memmove(mdata() + pos + sublen, data() + pos, orig_size - pos);
  memmove(mdata() + pos, str.data() + subpos, sublen);

  return *this;
}

inline tstring& tstring::insert(size_t pos, size_t n, char c) {
  size_t size_ = size();
  TF_TString_ResizeUninitialized(&tstr_, size_ + n);

  memmove(mdata() + pos + n, data() + pos, size_ - pos);
  memset(mdata() + pos, c, n);

  return *this;
}

inline void tstring::swap(tstring& str) {
  // TODO(dero): Invalid for OFFSET (unimplemented).
  std::swap(tstr_, str.tstr_);
}

inline void tstring::push_back(char ch) { append(1, ch); }

// Friends

inline bool operator==(const char* a, const tstring& b) {
  return ::strlen(a) == b.size() && ::memcmp(a, b.data(), b.size()) == 0;
}

inline bool operator==(const std::string& a, const tstring& b) {
  return a.size() == b.size() && ::memcmp(a.data(), b.data(), b.size()) == 0;
}

inline tstring operator+(const tstring& a, const tstring& b) {
  tstring r;
  r.reserve(a.size() + b.size());
  r.append(a);
  r.append(b);

  return r;
}

inline std::ostream& operator<<(std::ostream& o, const tstring& str) {
  return o.write(str.data(), str.size());
}

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_TSTRING_H_
