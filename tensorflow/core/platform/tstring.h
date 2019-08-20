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

#ifndef TENSORFLOW_CORE_PLATFORM_TSTRING_H_
#define TENSORFLOW_CORE_PLATFORM_TSTRING_H_

#include <string>

// TODO(b/138799229): Used to toggle until global presubmits pass.
// #define USE_TSTRING

#ifdef USE_TSTRING

// The inclusion of absl/strings/string_view.h in tstring.h would preclude the
// use of tstring in tflite.  Given that, in order to mitigate the forced
// inclusion of absl/strings/string_view.h while providing convenience methods
// for implicit conversion, we replace explicit uses of absl::string_view with a
// forward declaration and associated templates.
namespace absl {
class string_view;
}

namespace tensorflow {

// tensorflow::tstring is the scalar type for DT_STRING tensors.
//
// TODO(b/138799229): In order to ease migration from tensorflow::string to
// tensorflow::tstring, we define a simplified tstring class which wraps
// std::string.  The API defined below is the expected subset of methods for
// tstring.
//
// The underlying implementation of tstring will be replaced with the one
// defined in [1] once the migration in tensorflow/ is complete.
//
// [1] https://github.com/tensorflow/community/pull/91
class tstring {
  std::string str_;

  template <typename T, typename = void>
  struct ResizeUninitialized {
    static void Resize(T& s, size_t new_size) { s.resize(new_size); }
  };

  template <typename T>
  struct ResizeUninitialized<
      T, decltype(std::declval<T>().__resize_default_init(0))> {
    static void Resize(T& s, size_t new_size) {
      s.__resize_default_init(new_size);
    }
  };

 public:
  tstring() = default;

  tstring(const tstring&) = default;

  tstring(const std::string& str) : str_(str) {}

  tstring(const char* str, size_t len) : str_(str, len) {}

  tstring(const char* str) : str_(str) {}

  template <typename T,
            typename std::enable_if<std::is_same<T, absl::string_view>::value,
                                    T>::type* = nullptr>
  explicit tstring(const T& str) : str_(str.data(), str.size()) {}

  tstring(tstring&&) noexcept = default;

  ~tstring() = default;

  tstring& operator=(const tstring& str) = default;

  tstring& operator=(const std::string& str) {
    str_ = str;

    return *this;
  }

  template <typename T,
            typename std::enable_if<std::is_same<T, absl::string_view>::value,
                                    T>::type* = nullptr>
  tstring& operator=(const T& str) {
    str_.assign(str.data(), str.size());

    return *this;
  }

  tstring& operator=(const char* str) {
    str_ = str;

    return *this;
  }

  tstring& operator=(tstring&&) noexcept = default;

  bool operator<(const tstring& o) const { return str_ < o.str_; }

  bool operator>(const tstring& o) const { return str_ > o.str_; }

  bool operator==(const char* o) const { return str_ == o; }

  bool operator==(const tstring& o) const { return str_ == o.str_; }

  bool operator!=(const char* o) const { return str_ != o; }

  bool operator!=(const tstring& o) const { return str_ != o.str_; }

  operator std::string() const { return str_; }

  template <typename T,
            typename std::enable_if<std::is_same<T, absl::string_view>::value,
                                    T>::type* = nullptr>
  operator T() const {
    return T(str_.data(), str_.size());
  }

  bool empty() const { return str_.empty(); }

  size_t length() const { return str_.length(); }

  size_t size() const { return str_.size(); }

  const char* c_str() const { return str_.c_str(); }

  const char* data() const { return str_.data(); }

  const char& operator[](size_t i) const { return str_[i]; }

  char* data() { return &str_[0]; }

  char& operator[](size_t i) { return str_[i]; }

  void resize(size_t new_size) { str_.resize(new_size); }

  void resize_uninitialized(size_t new_size) {
    ResizeUninitialized<decltype(str_)>::Resize(str_, new_size);
  }

  tstring& assign(const char* str, size_t len) {
    str_.assign(str, len);

    return *this;
  }

  tstring& assign(const char* str) {
    str_.assign(str);

    return *this;
  }

  friend const tstring operator+(const tstring& a, const tstring& b);
  friend bool operator==(const char* a, const tstring& b);
  friend bool operator==(const std::string& a, const tstring& b);
  friend std::ostream& operator<<(std::ostream& o, const tstring& str);
  friend std::hash<tstring>;
};

inline bool operator==(const char* a, const tstring& b) { return a == b.str_; }

inline bool operator==(const std::string& a, const tstring& b) {
  return a == b.str_;
}

inline const tstring operator+(const tstring& a, const tstring& b) {
  return tstring(a.str_ + b.str_);
}

inline std::ostream& operator<<(std::ostream& o, const tstring& str) {
  return o << str.str_;
}

}  // namespace tensorflow

namespace std {
template <>
struct hash<tensorflow::tstring> {
  size_t operator()(const tensorflow::tstring& o) const {
    std::hash<std::string> fn;
    return fn(o.str_);
  }
};
}  // namespace std

#else  // USE_TSTRING

namespace tensorflow {

typedef std::string tstring;

}  // namespace tensorflow

#endif  // USE_TSTRING

#endif  // TENSORFLOW_CORE_PLATFORM_TSTRING_H_
