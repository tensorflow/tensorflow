/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBTF_IMPL_STRING_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBTF_IMPL_STRING_H_

#include <iosfwd>
#include <string>

namespace tf {
namespace libtf {
namespace impl {

/** A string value.
 *  This class wraps an interned, immutable string value. Currently, interned
 *  values are never deleted, so memory usage increases without bound as new
 *  strings are created.
 */
class String final {
 public:
  /** Interning constructor.
   * Interns the given string value.
   */
  explicit String(const char* s);

  String() : String("") {}
  String(const String& s) : value_(s.value_) {}

  // This is the same as the default equality operator, which works because
  // we're interning all strings. It is specified here so we are explicit about
  // it. We're not saying "= default;" because we can't use C++20 features yet.
  bool operator==(const String& other) const { return value_ == other.value_; }

  const std::string& str() const { return *value_; }

  /** Absl hash function. */
  template <typename H>
  friend H AbslHashValue(H h, const String& s) {
    return H::combine(std::move(h), *s.value_);
  }

 private:
  //! The interned string value. This is never null.
  const std::string* value_;
};

// This is defined in the `iostream.cc` file in this directory. It is not
// defined inline here because the `iosfwd` header does not provide enough
// functionality (in Windows), and we don't want to include `iostream` to avoid
// increasing the binary size.
std::ostream& operator<<(std::ostream& o, const String& str);

}  // namespace impl
}  // namespace libtf
}  // namespace tf

#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBTF_IMPL_STRING_H_
