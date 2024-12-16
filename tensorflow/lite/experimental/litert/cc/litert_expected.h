// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_EXPECTED_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_EXPECTED_H_

#include <initializer_list>
#include <memory>
#include <ostream>
#include <type_traits>
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"

namespace litert {

// An "Expected" incapsulates the result of some routine which may have an
// unexpected result. Unexpected results in this context are a standard
// LiteRtStatus plus extra usability data such as error messages. This is
// similar to an absl::StatusOr or std::expected (C++23) but better integrated
// with LiteRtStatus as the canonical status code.

// C++ wrapper around LiteRtStatus code. Provides a status as well
// as an error message.
class Error {
 public:
  // Construct Unexpected from status and optional error message. NOTE:
  // kLiteRtStatusOk should not be passed to Unexpected.
  explicit Error(LiteRtStatus status, absl::string_view message = "")
      : status_(status), message_(message) {
    ABSL_DCHECK(status != kLiteRtStatusOk);
  }

  // Get the status.
  constexpr LiteRtStatus Status() const { return status_; }

  // Get the error message, empty string if none was attached.
  constexpr absl::string_view Message() const { return message_; }

  friend std::ostream& operator<<(std::ostream& stream, const Error& error) {
    return stream << error.Message();
  }

 private:
  LiteRtStatus status_;
  absl::string_view message_;
};

class Unexpected {
 public:
  template <class... Args>
  constexpr explicit Unexpected(Args&&... args)
      : error_(std::forward<Args>(args)...) {}

  // Allow for implicit conversion from convertible Error value inplace.
  // NOLINTNEXTLINE
  Unexpected(class Error&& e) : error_(std::move(e)) {}

  Unexpected(Unexpected&& other) = default;
  Unexpected(const Unexpected& other) = default;
  Unexpected& operator=(Unexpected&& other) = default;
  Unexpected& operator=(const Unexpected& other) = default;

  constexpr const class Error& Error() const& noexcept { return error_; }
  constexpr class Error& Error() & noexcept { return error_; }
  constexpr const class Error&& Error() const&& noexcept {
    return std::move(error_);
  }
  constexpr class Error&& Error() && noexcept { return std::move(error_); }

 private:
  class Error error_;
};

// Utility for generic return values that may be a statused failure.
// Expecteds store and own the lifetime of either an Unexpected, or a T.
// T may be any type, primitive or non-primitive.
//
// No dynamic allocations occur during initialization,
// so the underlying T is only movable (as opposed to something like "release").
// Arguments should be constructed inplace at the time of initilizing
// the expcted if possible.
//
// Unexpected&& and T&& may be implicitly casted
// to an Expected. For example,
//
// Expected<Foo> Bar() {
//   bool success = ...
//   if (!success) { return Unexpected(kLiteRtStatus, "Bad Baz"); }
//   return Foo();
// }
//
template <class T>
class Expected {
 public:
  // Construct Expected with T inplace.

  // Construct T from initializer list inplace.
  template <class U>
  Expected(std::initializer_list<U> il) : has_value_(true), value_(il) {}

  // Construct T from forwarded args inplace.
  template <class... Args>
  explicit Expected(Args&&... args)
      : has_value_(true), value_(std::forward<Args>(args)...) {}

  // Allow for implicit conversion from convertible T value inplace.
  // NOLINTNEXTLINE
  Expected(const T& t) : has_value_(true), value_(t) {}
  // NOLINTNEXTLINE
  Expected(T&& t) : has_value_(true), value_(std::move(t)) {}

  // Construct from Unexpected inplace.

  // Allow for implicit conversion from Error.
  // NOLINTNEXTLINE
  Expected(const Unexpected& err) : has_value_(false), unexpected_(err) {}
  // NOLINTNEXTLINE
  Expected(Unexpected&& err) : has_value_(false), unexpected_(std::move(err)) {}
  // NOLINTNEXTLINE
  Expected(const class Error& e) : has_value_(false), unexpected_(e) {}

  // Copy/move

  Expected(Expected&& other) : has_value_(other.HasValue()) {
    if (HasValue()) {
      ConstructAt(std::addressof(value_), std::move(other.value_));
    } else {
      ConstructAt(std::addressof(unexpected_), std::move(other.unexpected_));
    }
  }

  Expected(const Expected& other) : has_value_(other.has_value_) {
    if (HasValue()) {
      ConstructAt(std::addressof(value_), other.value_);
      value_ = other.value_;
    } else {
      ConstructAt(std::addressof(unexpected_), other.unexpected_);
    }
  }

  Expected& operator=(Expected&& other) {
    if (this != &other) {
      Expected::~Expected();
      has_value_ = other.has_value_;
      if (HasValue()) {
        value_ = std::move(other.Value());
      } else {
        unexpected_ = std::move(other.unexpected_);
      }
    }
    return *this;
  }

  Expected& operator=(const Expected& other) {
    ~Expected();
    has_value_ = other.has_value_;
    if (HasValue()) {
      value_ = other.value_;
    } else {
      unexpected_ = other.unexpected_;
    }
    return *this;
  }

  ~Expected() {
    if (has_value_ && std::is_destructible<T>()) {
      value_.~T();
    } else {
      unexpected_.~Unexpected();
    }
  }

  // Observers for T value, program exits if it doesn't have one.
  const T& Value() const& {
    CheckVal();
    return value_;
  }

  T& Value() & {
    CheckVal();
    return value_;
  }

  const T&& Value() const&& {
    CheckVal();
    return std::move(value_);
  }

  T&& Value() && {
    CheckVal();
    return std::move(value_);
  }

  const T* operator->() const {
    CheckVal();
    return &value_;
  }

  T* operator->() {
    CheckVal();
    return &value_;
  }

  const T& operator*() const& { return Value(); }

  T& operator*() & { return Value(); }

  const T&& operator*() const&& { return std::move(Value()); }

  T&& operator*() && { return std::move(Value()); }

  // Observer for Unexpected, program exits if it doesn't have one.
  const class Error& Error() const& {
    CheckNoVal();
    return unexpected_.Error();
  }

  class Error& Error() & {
    CheckNoVal();
    return unexpected_.Error();
  }

  const class Error&& Error() const&& {
    CheckNoVal();
    return std::move(unexpected_.Error());
  }

  class Error&& Error() && {
    CheckNoVal();
    return std::move(unexpected_.Error());
  }

  // Does this expected contain a T Value. It contains an unexpected if not.
  bool HasValue() const { return has_value_; }

  // Convert to bool for HasValue.
  explicit operator bool() const { return HasValue(); }

 private:
  bool has_value_;
  union {
    T value_;
    Unexpected unexpected_;
  };
  void CheckNoVal() const { ABSL_CHECK(!HasValue()); }
  void CheckVal() const { ABSL_CHECK(HasValue()); }
};

template <>
class Expected<void> {
 public:
  // Implicit construction is used to simplify returning a valid value, e.g., in
  // "return {};"
  Expected() : has_value_(true) {}

  // Construct from Unexpected inplace.

  // Allow for implicit conversion from Error.
  // NOLINTNEXTLINE
  Expected(const Unexpected& err) : has_value_(false), unexpected_(err) {}
  // NOLINTNEXTLINE
  Expected(Unexpected&& err) : has_value_(false), unexpected_(std::move(err)) {}
  // NOLINTNEXTLINE
  Expected(const Error& e) : has_value_(false), unexpected_(e) {}

  ~Expected() {
    if (!has_value_) {
      unexpected_.~Unexpected();
    }
  }

  Expected& operator=(Expected&& other) {
    if (this != &other) {
      Expected::~Expected();
      has_value_ = other.has_value_;
      unexpected_ = std::move(other.unexpected_);
    }
    return *this;
  }

  Expected& operator=(const Expected& other) {
    if (this != &other) {
      Expected::~Expected();
      has_value_ = other.has_value_;
      unexpected_ = other.unexpected_;
    }
    return *this;
  }

  // Observer for Unexpected, program exits if it doesn't have one.
  const class Error& Error() const& {
    CheckNoVal();
    return unexpected_.Error();
  }

  class Error& Error() & {
    CheckNoVal();
    return unexpected_.Error();
  }

  const class Error&& Error() const&& {
    CheckNoVal();
    return std::move(unexpected_.Error());
  }

  class Error&& Error() && {
    CheckNoVal();
    return std::move(unexpected_.Error());
  }

  // Does this expected contain a T Value. It contains an unexpected if not.
  bool HasValue() const { return has_value_; }

  // Convert to bool for HasValue.
  explicit operator bool() const { return HasValue(); }

 private:
  bool has_value_;
  union {
    Unexpected unexpected_;
  };
  void CheckNoVal() const { ABSL_CHECK(!HasValue()); }
  void CheckVal() const { ABSL_CHECK(HasValue()); }
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_CC_LITERT_EXPECTED_H_
