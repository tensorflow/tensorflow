/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LIB_GTL_OPTIONAL_H_
#define TENSORFLOW_LIB_GTL_OPTIONAL_H_

#include <functional>
#include <initializer_list>
#include <type_traits>
#include <utility>

#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace gtl {

// A value of type gtl::optional<T> holds either a value of T or an
// "empty" value.  When it holds a value of T, it stores it as a direct
// subobject, so sizeof(optional<T>) is approximately sizeof(T)+1. The interface
// is based on the upcoming std::optional<T>, and gtl::optional<T> is
// designed to be cheaply drop-in replaceable by std::optional<T>, once it is
// rolled out.
//
// This implementation is based on the specification in N4606 Section 20.6:
//    http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/n4606.pdf
//
// Differences between gtl::optional<T> and std::optional<T> include:
//    - gtl::optional<T> is basically a proper subset of
//         std::optional<T>.
//    - constexpr not used. (dependency on some differences between C++11 and
//         C++14.)
//    - noexcept not used.
//    - exceptions not used - in lieu of exceptions we use CHECK-failure.
//    - (b/30115483) gtl::make_optional has a different API than
//      std::make_optional.
//
// (b/30115368) std::optional<T> might not quite be a drop-in replacement for
// std::experimental::optional<T> because the semantics of relational operators
// are slightly different. The best way of making sure you aren't affected by
// those changes is to make sure that your type T defines all of the operators
// consistently. (x <= y is exactly equivalent to !(x > y), etc.)
//
// Synopsis:
//
//     #include "tensorflow/core/lib/gtl/optional.h"
//
//     tensorflow::gtl::optional<string> f() {
//       string result;
//       if (...) {
//          ...
//          result = ...;
//          return result;
//       } else {
//          ...
//          return tensorflow::gtl::nullopt;
//       }
//     }
//
//     int main() {
//         tensorflow::gtl::optional<string> optstr = f();
//         if (optstr) {
//            // non-empty
//            print(optstr.value());
//         } else {
//            // empty
//            error();
//         }
//     }
template <typename T>
class optional;

// The tag constant `in_place` is used as the first parameter of an optional<T>
// constructor to indicate that the remaining arguments should be forwarded
// to the underlying T constructor.
struct in_place_t {};
extern const in_place_t in_place;

// The tag constant `nullopt` is used to indicate an empty optional<T> in
// certain functions, such as construction or assignment.
struct nullopt_t {
  // It must not be default-constructible to avoid ambiguity for opt = {}.
  explicit constexpr nullopt_t(int /*unused*/) {}
};
extern const nullopt_t nullopt;

// See comment above first declaration.
template <typename T>
class optional {
 public:
  typedef T value_type;

  // A default constructed optional holds the empty value, NOT a default
  // constructed T.
  optional() {}

  // An optional initialized with `nullopt` holds the empty value.
  optional(nullopt_t /*unused*/) {}  // NOLINT(runtime/explicit)

  // Copy constructor, standard semantics.
  optional(const optional& src) {
    if (src) {
      construct(src.reference());
    }
  }

  // Move constructor, standard semantics.
  optional(optional&& src) noexcept(
      std::is_nothrow_move_constructible<T>::value) {
    if (src) {
      construct(std::move(src.reference()));
    }
  }

  // Creates a non-empty optional<T> with a copy of the given value of T.
  optional(const T& src) {  // NOLINT(runtime/explicit)
    construct(src);
  }

  // Creates a non-empty optional<T> with a moved-in value of T.
  optional(T&& src) {  // NOLINT
    construct(std::move(src));
  }

  // optional<T>(in_place, arg1, arg2, arg3) constructs a non-empty optional
  // with an in-place constructed value of T(arg1,arg2,arg3).
  template <typename... Args>
  explicit optional(in_place_t /*unused*/,
                    Args&&... args) {  // NOLINT(build/c++11)
    construct(std::forward<Args>(args)...);
  }

  // optional<T>(in_place, {arg1, arg2, arg3}) constructs a non-empty optional
  // with an in-place list-initialized value of T({arg1, arg2, arg3}).
  template <class U, typename... Args>
  explicit optional(in_place_t /*unused*/, std::initializer_list<U> il,
                    Args&&... args) {  // NOLINT(build/c++11)
    construct(il, std::forward<Args>(args)...);
  }

  // Destructor, standard semantics.
  ~optional() { reset(); }

  // Assignment from nullopt: opt = nullopt
  optional& operator=(nullopt_t /*unused*/) {
    reset();
    return *this;
  }

  // Copy assigment, standard semantics.
  optional& operator=(const optional& src) {
    if (src) {
      operator=(src.reference());
    } else {
      reset();
    }
    return *this;
  }

  // Move assignment, standard semantics.
  optional& operator=(optional&& src) {  // NOLINT(build/c++11)
    if (src) {
      operator=(std::move(src.reference()));
    } else {
      reset();
    }
    return *this;
  }

  // Copy assigment from T.  If empty becomes copy construction.
  optional& operator=(const T& src) {  // NOLINT(build/c++11)
    if (*this) {
      reference() = src;
    } else {
      construct(src);
    }
    return *this;
  }

  // Move assignment from T.  If empty becomes move construction.
  optional& operator=(T&& src) {  // NOLINT(build/c++11)
    if (*this) {
      reference() = std::move(src);
    } else {
      construct(std::move(src));
    }
    return *this;
  }

  // Destroys the inner T value if one is present.
  void reset() {
    if (engaged_) {
      destruct();
    }
    DCHECK(!engaged_);
  }

  // Emplace reconstruction.  (Re)constructs the underlying T in-place with the
  // given arguments forwarded:
  //
  // optional<Foo> opt;
  // opt.emplace(arg1,arg2,arg3);  (Constructs Foo(arg1,arg2,arg3))
  //
  // If the optional is non-empty, and the `args` refer to subobjects of the
  // current object, then behaviour is undefined.  This is because the current
  // object will be destructed before the new object is constructed with `args`.
  //
  template <typename... Args>
  void emplace(Args&&... args) {
    reset();
    construct(std::forward<Args>(args)...);
  }

  // Emplace reconstruction with initializer-list.  See immediately above.
  template <class U, class... Args>
  void emplace(std::initializer_list<U> il, Args&&... args) {
    reset();
    construct(il, std::forward<Args>(args)...);
  }

  // Swap, standard semantics.
  void swap(optional& src) {
    if (*this) {
      if (src) {
        using std::swap;
        swap(reference(), src.reference());
      } else {
        src.construct(std::move(reference()));
        destruct();
      }
    } else {
      if (src) {
        construct(std::move(src.reference()));
        src.destruct();
      } else {
        // no effect (swap(disengaged, disengaged))
      }
    }
  }

  // You may use `*opt`, and `opt->m`, to access the underlying T value and T's
  // member `m`, respectively.  If the optional is empty, behaviour is
  // undefined.
  const T* operator->() const {
    DCHECK(engaged_);
    return pointer();
  }
  T* operator->() {
    DCHECK(engaged_);
    return pointer();
  }
  const T& operator*() const & {
    DCHECK(engaged_);
    return reference();
  }
  T& operator*() & {
    DCHECK(engaged_);
    return reference();
  }
  const T&& operator*() const && {
    DCHECK(engaged_);
    return std::move(reference());
  }
  T&& operator*() && {
    DCHECK(engaged_);
    return std::move(reference());
  }

  // In a bool context an optional<T> will return false if and only if it is
  // empty.
  //
  //   if (opt) {
  //     // do something with opt.value();
  //   } else {
  //     // opt is empty
  //   }
  //
  explicit operator bool() const { return engaged_; }

  // Returns false if and only if *this is empty.
  bool has_value() const { return engaged_; }

  // Use `opt.value()` to get a reference to underlying value.  The constness
  // and lvalue/rvalue-ness of `opt` is preserved to the view of the T
  // subobject.
  const T& value() const & {
    CHECK(*this) << "Bad optional access";
    return reference();
  }
  T& value() & {
    CHECK(*this) << "Bad optional access";
    return reference();
  }
  T&& value() && {  // NOLINT(build/c++11)
    CHECK(*this) << "Bad optional access";
    return std::move(reference());
  }
  const T&& value() const && {  // NOLINT(build/c++11)
    CHECK(*this) << "Bad optional access";
    return std::move(reference());
  }

  // Use `opt.value_or(val)` to get either the value of T or the given default
  // `val` in the empty case.
  template <class U>
  T value_or(U&& val) const & {
    if (*this) {
      return reference();
    } else {
      return static_cast<T>(std::forward<U>(val));
    }
  }
  template <class U>
  T value_or(U&& val) && {  // NOLINT(build/c++11)
    if (*this) {
      return std::move(reference());
    } else {
      return static_cast<T>(std::forward<U>(val));
    }
  }

 private:
  // Private accessors for internal storage viewed as pointer or reference to T.
  const T* pointer() const {
    return static_cast<const T*>(static_cast<const void*>(&storage_));
  }
  T* pointer() { return static_cast<T*>(static_cast<void*>(&storage_)); }
  const T& reference() const { return *pointer(); }
  T& reference() { return *pointer(); }

  // Construct inner T in place with given `args`.
  // Precondition: engaged_ is false
  // Postcondition: engaged_ is true
  template <class... Args>
  void construct(Args&&... args) {
    DCHECK(!engaged_);
    engaged_ = true;
    new (pointer()) T(std::forward<Args>(args)...);
    DCHECK(engaged_);
  }

  // Destruct inner T.
  // Precondition: engaged_ is true
  // Postcondition: engaged_ is false
  void destruct() {
    DCHECK(engaged_);
    pointer()->T::~T();
    engaged_ = false;
    DCHECK(!engaged_);
  }

  // The internal storage for a would-be T value, constructed and destroyed
  // with placement new and placement delete.
  typename std::aligned_storage<sizeof(T), alignof(T)>::type storage_;

  // Whether or not this optional is non-empty.
  bool engaged_ = false;

  // T constaint checks.  You can't have an optional of nullopt_t, in_place_t or
  // a reference.
  static_assert(
      !std::is_same<nullopt_t, typename std::remove_cv<T>::type>::value,
      "optional<nullopt_t> is not allowed.");
  static_assert(
      !std::is_same<in_place_t, typename std::remove_cv<T>::type>::value,
      "optional<in_place_t> is not allowed.");
  static_assert(!std::is_reference<T>::value,
                "optional<reference> is not allowed.");
};

// make_optional(v) creates a non-empty optional<T> where the type T is deduced
// from v.  Can also be explicitly instantiated as make_optional<T>(v).
template <typename T>
optional<typename std::decay<T>::type> make_optional(T&& v) {
  return optional<typename std::decay<T>::type>(std::forward<T>(v));
}

// Relational operators. Empty optionals are considered equal to each
// other and less than non-empty optionals. Supports relations between
// optional<T> and optional<T>, between optional<T> and T, and between
// optional<T> and nullopt.
// Note: We're careful to support T having non-bool relationals.

// Relational operators [optional.relops]
// The C++17 (N4606) "Returns:" statements are translated into code
// in an obvious way here, and the original text retained as function docs.
// Returns: If bool(x) != bool(y), false; otherwise if bool(x) == false, true;
// otherwise *x == *y.
template <class T>
constexpr bool operator==(const optional<T>& x, const optional<T>& y) {
  return static_cast<bool>(x) != static_cast<bool>(y)
             ? false
             : static_cast<bool>(x) == false ? true : *x == *y;
}
// Returns: If bool(x) != bool(y), true; otherwise, if bool(x) == false, false;
// otherwise *x != *y.
template <class T>
constexpr bool operator!=(const optional<T>& x, const optional<T>& y) {
  return static_cast<bool>(x) != static_cast<bool>(y)
             ? true
             : static_cast<bool>(x) == false ? false : *x != *y;
}
// Returns: If !y, false; otherwise, if !x, true; otherwise *x < *y.
template <class T>
constexpr bool operator<(const optional<T>& x, const optional<T>& y) {
  return !y ? false : !x ? true : *x < *y;
}
// Returns: If !x, false; otherwise, if !y, true; otherwise *x > *y.
template <class T>
constexpr bool operator>(const optional<T>& x, const optional<T>& y) {
  return !x ? false : !y ? true : *x > *y;
}
// Returns: If !x, true; otherwise, if !y, false; otherwise *x <= *y.
template <class T>
constexpr bool operator<=(const optional<T>& x, const optional<T>& y) {
  return !x ? true : !y ? false : *x <= *y;
}
// Returns: If !y, true; otherwise, if !x, false; otherwise *x >= *y.
template <class T>
constexpr bool operator>=(const optional<T>& x, const optional<T>& y) {
  return !y ? true : !x ? false : *x >= *y;
}

// Comparison with nullopt [optional.nullops]
// The C++17 (N4606) "Returns:" statements are used directly here.
template <class T>
constexpr bool operator==(const optional<T>& x, nullopt_t) noexcept {
  return !x;
}
template <class T>
constexpr bool operator==(nullopt_t, const optional<T>& x) noexcept {
  return !x;
}
template <class T>
constexpr bool operator!=(const optional<T>& x, nullopt_t) noexcept {
  return static_cast<bool>(x);
}
template <class T>
constexpr bool operator!=(nullopt_t, const optional<T>& x) noexcept {
  return static_cast<bool>(x);
}
template <class T>
constexpr bool operator<(const optional<T>& x, nullopt_t) noexcept {
  return false;
}
template <class T>
constexpr bool operator<(nullopt_t, const optional<T>& x) noexcept {
  return static_cast<bool>(x);
}
template <class T>
constexpr bool operator<=(const optional<T>& x, nullopt_t) noexcept {
  return !x;
}
template <class T>
constexpr bool operator<=(nullopt_t, const optional<T>& x) noexcept {
  return true;
}
template <class T>
constexpr bool operator>(const optional<T>& x, nullopt_t) noexcept {
  return static_cast<bool>(x);
}
template <class T>
constexpr bool operator>(nullopt_t, const optional<T>& x) noexcept {
  return false;
}
template <class T>
constexpr bool operator>=(const optional<T>& x, nullopt_t) noexcept {
  return true;
}
template <class T>
constexpr bool operator>=(nullopt_t, const optional<T>& x) noexcept {
  return !x;
}

// Comparison with T [optional.comp_with_t]
// The C++17 (N4606) "Equivalent to:" statements are used directly here.
template <class T>
constexpr bool operator==(const optional<T>& x, const T& v) {
  return static_cast<bool>(x) ? *x == v : false;
}
template <class T>
constexpr bool operator==(const T& v, const optional<T>& x) {
  return static_cast<bool>(x) ? v == *x : false;
}
template <class T>
constexpr bool operator!=(const optional<T>& x, const T& v) {
  return static_cast<bool>(x) ? *x != v : true;
}
template <class T>
constexpr bool operator!=(const T& v, const optional<T>& x) {
  return static_cast<bool>(x) ? v != *x : true;
}
template <class T>
constexpr bool operator<(const optional<T>& x, const T& v) {
  return static_cast<bool>(x) ? *x < v : true;
}
template <class T>
constexpr bool operator<(const T& v, const optional<T>& x) {
  return static_cast<bool>(x) ? v < *x : false;
}
template <class T>
constexpr bool operator<=(const optional<T>& x, const T& v) {
  return static_cast<bool>(x) ? *x <= v : true;
}
template <class T>
constexpr bool operator<=(const T& v, const optional<T>& x) {
  return static_cast<bool>(x) ? v <= *x : false;
}
template <class T>
constexpr bool operator>(const optional<T>& x, const T& v) {
  return static_cast<bool>(x) ? *x > v : false;
}
template <class T>
constexpr bool operator>(const T& v, const optional<T>& x) {
  return static_cast<bool>(x) ? v > *x : true;
}
template <class T>
constexpr bool operator>=(const optional<T>& x, const T& v) {
  return static_cast<bool>(x) ? *x >= v : false;
}
template <class T>
constexpr bool operator>=(const T& v, const optional<T>& x) {
  return static_cast<bool>(x) ? v >= *x : true;
}

// Swap, standard semantics.
template <typename T>
void swap(optional<T>& a, optional<T>& b) {
  a.swap(b);
}

}  // namespace gtl
}  // namespace tensorflow

namespace std {

// std::hash specialization for gtl::optional.  Normally std::hash
// specializations are banned in Google code, but the arbiters granted a
// styleguide exception for this one in cl/95369397, as optional is following
// a standard library component.
template <class T>
struct hash<::tensorflow::gtl::optional<T>> {
  size_t operator()(const ::tensorflow::gtl::optional<T>& opt) const {
    if (opt) {
      return hash<T>()(*opt);
    } else {
      return static_cast<size_t>(0x297814aaad196e6dULL);
    }
  }
};

}  // namespace std

#endif  // TENSORFLOW_LIB_GTL_OPTIONAL_H_
