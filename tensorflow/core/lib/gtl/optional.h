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

#include <assert.h>
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
// This implementation is based on the specification in the latest draft as of
// 2017-01-05, section 20.6.
//
// Differences between gtl::optional<T> and std::optional<T> include:
//    - constexpr not used for nonconst member functions.
//      (dependency on some differences between C++11 and C++14.)
//    - nullopt and in_place are not constexpr. We need the inline variable
//      support in C++17 for external linkage.
//    - CHECK instead of throwing std::bad_optional_access.
//    - optional::swap() and swap() relies on std::is_(nothrow_)swappable
//      which is introduced in C++17. So we assume is_swappable is always true
//      and is_nothrow_swappable is same as std::is_trivial.
//    - make_optional cannot be constexpr due to absence of guaranteed copy
//      elision.
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
  struct init_t {};
  static init_t init;
  // It must not be default-constructible to avoid ambiguity for opt = {}.
  // Note the non-const reference, it is to eliminate ambiguity for code like:
  // struct S { int value; };
  //
  // void Test() {
  //   optional<S> opt;
  //   opt = {{}};
  // }
  explicit constexpr nullopt_t(init_t& /*unused*/) {}  // NOLINT
};
extern const nullopt_t nullopt;

namespace internal_optional {

// define forward locally because std::forward is not constexpr until C++14
template <typename T>
constexpr T&& forward(typename std::remove_reference<T>::type&
                          t) noexcept {  // NOLINT(runtime/references)
  return static_cast<T&&>(t);
}

struct empty_struct {};
// This class stores the data in optional<T>.
// It is specialized based on whether T is trivially destructible.
// This is the specialization for non trivially destructible type.
template <typename T, bool = std::is_trivially_destructible<T>::value>
class optional_data_dtor_base {
 protected:
  // Whether there is data or not.
  bool engaged_;
  // data storage
  union {
    empty_struct dummy_;
    T data_;
  };

  void destruct() noexcept {
    if (engaged_) {
      data_.~T();
      engaged_ = false;
    }
  }

  // dummy_ must be initialized for constexpr constructor
  constexpr optional_data_dtor_base() noexcept : engaged_(false), dummy_{} {}

  template <typename... Args>
  constexpr explicit optional_data_dtor_base(in_place_t, Args&&... args)
      : engaged_(true), data_(internal_optional::forward<Args>(args)...) {}

  ~optional_data_dtor_base() { destruct(); }
};

// Specialization for trivially destructible type.
template <typename T>
class optional_data_dtor_base<T, true> {
 protected:
  // Whether there is data or not.
  bool engaged_;
  // data storage
  union {
    empty_struct dummy_;
    T data_;
  };
  void destruct() noexcept { engaged_ = false; }

  // dummy_ must be initialized for constexpr constructor
  constexpr optional_data_dtor_base() noexcept : engaged_(false), dummy_{} {}

  template <typename... Args>
  constexpr explicit optional_data_dtor_base(in_place_t, Args&&... args)
      : engaged_(true), data_(internal_optional::forward<Args>(args)...) {}

  ~optional_data_dtor_base() = default;
};

template <typename T>
class optional_data : public optional_data_dtor_base<T> {
 protected:
  using base = optional_data_dtor_base<T>;
  using base::base;

  T* pointer() { return &this->data_; }

  constexpr const T* pointer() const { return &this->data_; }

  template <typename... Args>
  void construct(Args&&... args) {
    new (pointer()) T(std::forward<Args>(args)...);
    this->engaged_ = true;
  }

  template <typename U>
  void assign(U&& u) {
    if (this->engaged_) {
      this->data_ = std::forward<U>(u);
    } else {
      construct(std::forward<U>(u));
    }
  }

  optional_data() = default;

  optional_data(const optional_data& rhs) {
    if (rhs.engaged_) {
      construct(rhs.data_);
    }
  }

  optional_data(optional_data&& rhs) noexcept(
      std::is_nothrow_move_constructible<T>::value) {
    if (rhs.engaged_) {
      construct(std::move(rhs.data_));
    }
  }

  optional_data& operator=(const optional_data& rhs) {
    if (rhs.engaged_) {
      assign(rhs.data_);
    } else {
      this->destruct();
    }
    return *this;
  }

  optional_data& operator=(optional_data&& rhs) noexcept(
      std::is_nothrow_move_assignable<T>::value&&
          std::is_nothrow_move_constructible<T>::value) {
    if (rhs.engaged_) {
      assign(std::move(rhs.data_));
    } else {
      this->destruct();
    }
    return *this;
  }
};

// ordered by level of restriction, from low to high.
// copyable implies movable.
enum class copy_traits { copyable = 0, movable = 1, non_movable = 2 };

// base class for enabling/disabling copy/move constructor.
template <copy_traits>
class optional_ctor_base;

template <>
class optional_ctor_base<copy_traits::copyable> {
 public:
  constexpr optional_ctor_base() = default;
  optional_ctor_base(const optional_ctor_base&) = default;
  optional_ctor_base(optional_ctor_base&&) = default;
  optional_ctor_base& operator=(const optional_ctor_base&) = default;
  optional_ctor_base& operator=(optional_ctor_base&&) = default;
};

template <>
class optional_ctor_base<copy_traits::movable> {
 public:
  constexpr optional_ctor_base() = default;
  optional_ctor_base(const optional_ctor_base&) = delete;
  optional_ctor_base(optional_ctor_base&&) = default;
  optional_ctor_base& operator=(const optional_ctor_base&) = default;
  optional_ctor_base& operator=(optional_ctor_base&&) = default;
};

template <>
class optional_ctor_base<copy_traits::non_movable> {
 public:
  constexpr optional_ctor_base() = default;
  optional_ctor_base(const optional_ctor_base&) = delete;
  optional_ctor_base(optional_ctor_base&&) = delete;
  optional_ctor_base& operator=(const optional_ctor_base&) = default;
  optional_ctor_base& operator=(optional_ctor_base&&) = default;
};

// base class for enabling/disabling copy/move assignment.
template <copy_traits>
class optional_assign_base;

template <>
class optional_assign_base<copy_traits::copyable> {
 public:
  constexpr optional_assign_base() = default;
  optional_assign_base(const optional_assign_base&) = default;
  optional_assign_base(optional_assign_base&&) = default;
  optional_assign_base& operator=(const optional_assign_base&) = default;
  optional_assign_base& operator=(optional_assign_base&&) = default;
};

template <>
class optional_assign_base<copy_traits::movable> {
 public:
  constexpr optional_assign_base() = default;
  optional_assign_base(const optional_assign_base&) = default;
  optional_assign_base(optional_assign_base&&) = default;
  optional_assign_base& operator=(const optional_assign_base&) = delete;
  optional_assign_base& operator=(optional_assign_base&&) = default;
};

template <>
class optional_assign_base<copy_traits::non_movable> {
 public:
  constexpr optional_assign_base() = default;
  optional_assign_base(const optional_assign_base&) = default;
  optional_assign_base(optional_assign_base&&) = default;
  optional_assign_base& operator=(const optional_assign_base&) = delete;
  optional_assign_base& operator=(optional_assign_base&&) = delete;
};

template <typename T>
constexpr copy_traits get_ctor_copy_traits() {
  return std::is_copy_constructible<T>::value
             ? copy_traits::copyable
             : std::is_move_constructible<T>::value ? copy_traits::movable
                                                    : copy_traits::non_movable;
}

template <typename T>
constexpr copy_traits get_assign_copy_traits() {
  return std::is_copy_assignable<T>::value &&
                 std::is_copy_constructible<T>::value
             ? copy_traits::copyable
             : std::is_move_assignable<T>::value &&
                       std::is_move_constructible<T>::value
                   ? copy_traits::movable
                   : copy_traits::non_movable;
}

// Whether T is constructible or convertible from optional<U>.
template <typename T, typename U>
struct is_constructible_convertible_from_optional
    : std::integral_constant<
          bool, std::is_constructible<T, optional<U>&>::value ||
                    std::is_constructible<T, optional<U>&&>::value ||
                    std::is_constructible<T, const optional<U>&>::value ||
                    std::is_constructible<T, const optional<U>&&>::value ||
                    std::is_convertible<optional<U>&, T>::value ||
                    std::is_convertible<optional<U>&&, T>::value ||
                    std::is_convertible<const optional<U>&, T>::value ||
                    std::is_convertible<const optional<U>&&, T>::value> {};

// Whether T is constructible or convertible or assignable from optional<U>.
template <typename T, typename U>
struct is_constructible_convertible_assignable_from_optional
    : std::integral_constant<
          bool, is_constructible_convertible_from_optional<T, U>::value ||
                    std::is_assignable<T&, optional<U>&>::value ||
                    std::is_assignable<T&, optional<U>&&>::value ||
                    std::is_assignable<T&, const optional<U>&>::value ||
                    std::is_assignable<T&, const optional<U>&&>::value> {};

}  // namespace internal_optional

template <typename T>
class optional : private internal_optional::optional_data<T>,
                 private internal_optional::optional_ctor_base<
                     internal_optional::get_ctor_copy_traits<T>()>,
                 private internal_optional::optional_assign_base<
                     internal_optional::get_assign_copy_traits<T>()> {
  using data_base = internal_optional::optional_data<T>;

 public:
  typedef T value_type;

  // [optional.ctor], constructors

  // A default constructed optional holds the empty value, NOT a default
  // constructed T.
  constexpr optional() noexcept {}

  // An optional initialized with `nullopt` holds the empty value.
  constexpr optional(nullopt_t) noexcept {}  // NOLINT(runtime/explicit)

  // Copy constructor, standard semantics.
  optional(const optional& src) = default;

  // Move constructor, standard semantics.
  optional(optional&& src) = default;

  // optional<T>(in_place, arg1, arg2, arg3) constructs a non-empty optional
  // with an in-place constructed value of T(arg1,arg2,arg3).
  // TODO(b/34201852): Add std::is_constructible<T, Args&&...> SFINAE.
  template <typename... Args>
  constexpr explicit optional(in_place_t, Args&&... args)
      : data_base(in_place_t(), internal_optional::forward<Args>(args)...) {}

  // optional<T>(in_place, {arg1, arg2, arg3}) constructs a non-empty optional
  // with an in-place list-initialized value of T({arg1, arg2, arg3}).
  template <typename U, typename... Args,
            typename = typename std::enable_if<std::is_constructible<
                T, std::initializer_list<U>&, Args&&...>::value>::type>
  constexpr explicit optional(in_place_t, std::initializer_list<U> il,
                              Args&&... args)
      : data_base(in_place_t(), il, internal_optional::forward<Args>(args)...) {
  }

  template <
      typename U = T,
      typename std::enable_if<
          std::is_constructible<T, U&&>::value &&
              !std::is_same<in_place_t, typename std::decay<U>::type>::value &&
              !std::is_same<optional<T>, typename std::decay<U>::type>::value &&
              std::is_convertible<U&&, T>::value,
          bool>::type = false>
  constexpr optional(U&& v)  // NOLINT
      : data_base(in_place_t(), internal_optional::forward<U>(v)) {}

  template <
      typename U = T,
      typename std::enable_if<
          std::is_constructible<T, U&&>::value &&
              !std::is_same<in_place_t, typename std::decay<U>::type>::value &&
              !std::is_same<optional<T>, typename std::decay<U>::type>::value &&
              !std::is_convertible<U&&, T>::value,
          bool>::type = false>
  explicit constexpr optional(U&& v)
      : data_base(in_place_t(), internal_optional::forward<U>(v)) {}

  // Converting copy constructor (implicit)
  template <
      typename U,
      typename std::enable_if<
          std::is_constructible<T, const U&>::value &&
              !internal_optional::is_constructible_convertible_from_optional<
                  T, U>::value &&
              std::is_convertible<const U&, T>::value,
          bool>::type = false>
  optional(const optional<U>& rhs) {  // NOLINT
    if (rhs) {
      this->construct(*rhs);
    }
  }

  // Converting copy constructor (explicit)
  template <
      typename U,
      typename std::enable_if<
          std::is_constructible<T, const U&>::value &&
              !internal_optional::is_constructible_convertible_from_optional<
                  T, U>::value &&
              !std::is_convertible<const U&, T>::value,
          bool>::type = false>
  explicit optional(const optional<U>& rhs) {
    if (rhs) {
      this->construct(*rhs);
    }
  }

  // Converting move constructor (implicit)
  template <
      typename U,
      typename std::enable_if<
          std::is_constructible<T, U&&>::value &&
              !internal_optional::is_constructible_convertible_from_optional<
                  T, U>::value &&
              std::is_convertible<U&&, T>::value,
          bool>::type = false>
  optional(optional<U>&& rhs) {  // NOLINT
    if (rhs) {
      this->construct(std::move(*rhs));
    }
  }

  // Converting move constructor (explicit)
  template <
      typename U,
      typename std::enable_if<
          std::is_constructible<T, U&&>::value &&
              !internal_optional::is_constructible_convertible_from_optional<
                  T, U>::value &&
              !std::is_convertible<U&&, T>::value,
          bool>::type = false>
  explicit optional(optional<U>&& rhs) {
    if (rhs) {
      this->construct(std::move(*rhs));
    }
  }

  // [optional.dtor], destructor, trivial if T is trivially destructible.
  ~optional() = default;

  // [optional.assign], assignment

  // Assignment from nullopt: opt = nullopt
  optional& operator=(nullopt_t) noexcept {
    this->destruct();
    return *this;
  }

  // Copy assigment, standard semantics.
  optional& operator=(const optional& src) = default;

  // Move assignment, standard semantics.
  optional& operator=(optional&& src) = default;

  // Value assignment
  template <
      typename U = T,
      typename = typename std::enable_if<
          !std::is_same<optional<T>, typename std::decay<U>::type>::value &&
          (!std::is_scalar<T>::value ||
           !std::is_same<T, typename std::decay<U>::type>::value) &&
          std::is_constructible<T, U>::value &&
          std::is_assignable<T&, U>::value>::type>
  optional& operator=(U&& v) {
    this->assign(std::forward<U>(v));
    return *this;
  }

  template <typename U,
            typename = typename std::enable_if<
                std::is_constructible<T, const U&>::value &&
                std::is_assignable<T&, const U&>::value &&
                !internal_optional::
                    is_constructible_convertible_assignable_from_optional<
                        T, U>::value>::type>
  optional& operator=(const optional<U>& rhs) {
    if (rhs) {
      this->assign(*rhs);
    } else {
      this->destruct();
    }
    return *this;
  }

  template <typename U,
            typename = typename std::enable_if<
                std::is_constructible<T, U>::value &&
                std::is_assignable<T&, U>::value &&
                !internal_optional::
                    is_constructible_convertible_assignable_from_optional<
                        T, U>::value>::type>
  optional& operator=(optional<U>&& rhs) {
    if (rhs) {
      this->assign(std::move(*rhs));
    } else {
      this->destruct();
    }
    return *this;
  }

  // [optional.mod], modifiers
  // Destroys the inner T value if one is present.
  void reset() noexcept { this->destruct(); }

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
  template <typename... Args,
            typename = typename std::enable_if<
                std::is_constructible<T, Args&&...>::value>::type>
  void emplace(Args&&... args) {
    this->destruct();
    this->construct(std::forward<Args>(args)...);
  }

  // Emplace reconstruction with initializer-list.  See immediately above.
  template <class U, class... Args,
            typename = typename std::enable_if<std::is_constructible<
                T, std::initializer_list<U>&, Args&&...>::value>::type>
  void emplace(std::initializer_list<U> il, Args&&... args) {
    this->destruct();
    this->construct(il, std::forward<Args>(args)...);
  }

  // [optional.swap], swap
  // Swap, standard semantics.
  void swap(optional& rhs) noexcept(
      std::is_nothrow_move_constructible<T>::value&&
          std::is_trivial<T>::value) {
    if (*this) {
      if (rhs) {
        using std::swap;
        swap(**this, *rhs);
      } else {
        rhs.construct(std::move(**this));
        this->destruct();
      }
    } else {
      if (rhs) {
        this->construct(std::move(*rhs));
        rhs.destruct();
      } else {
        // no effect (swap(disengaged, disengaged))
      }
    }
  }

  // [optional.observe], observers
  // You may use `*opt`, and `opt->m`, to access the underlying T value and T's
  // member `m`, respectively.  If the optional is empty, behaviour is
  // undefined.
  constexpr const T* operator->() const { return this->pointer(); }
  T* operator->() {
    assert(this->engaged_);
    return this->pointer();
  }
  constexpr const T& operator*() const & { return reference(); }
  T& operator*() & {
    assert(this->engaged_);
    return reference();
  }
  constexpr const T&& operator*() const && { return std::move(reference()); }
  T&& operator*() && {
    assert(this->engaged_);
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
  constexpr explicit operator bool() const noexcept { return this->engaged_; }

  // Returns false if and only if *this is empty.
  constexpr bool has_value() const noexcept { return this->engaged_; }

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
  constexpr T value_or(U&& v) const & {
    return static_cast<bool>(*this) ? **this
                                    : static_cast<T>(std::forward<U>(v));
  }
  template <class U>
  T value_or(U&& v) && {  // NOLINT(build/c++11)
    return static_cast<bool>(*this) ? std::move(**this)
                                    : static_cast<T>(std::forward<U>(v));
  }

 private:
  // Private accessors for internal storage viewed as reference to T.
  constexpr const T& reference() const { return *this->pointer(); }
  T& reference() { return *(this->pointer()); }

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

// [optional.specalg]
// Swap, standard semantics.
// This function shall not participate in overload resolution unless
// is_move_constructible_v<T> is true and is_swappable_v<T> is true.
// NOTE: we assume is_swappable is always true. There will be a compiling error
// if T is actually not Swappable.
template <typename T,
          typename std::enable_if<std::is_move_constructible<T>::value,
                                  bool>::type = false>
void swap(optional<T>& a, optional<T>& b) noexcept(noexcept(a.swap(b))) {
  a.swap(b);
}

// NOTE: make_optional cannot be constexpr in C++11 because the copy/move
// constructor is not constexpr and we don't have guaranteed copy elision
// util C++17. But they are still declared constexpr for consistency with
// the standard.

// make_optional(v) creates a non-empty optional<T> where the type T is deduced
// from v.  Can also be explicitly instantiated as make_optional<T>(v).
template <typename T>
constexpr optional<typename std::decay<T>::type> make_optional(T&& v) {
  return optional<typename std::decay<T>::type>(std::forward<T>(v));
}

template <typename T, typename... Args>
constexpr optional<T> make_optional(Args&&... args) {
  return optional<T>(in_place_t(), internal_optional::forward<Args>(args)...);
}

template <typename T, typename U, typename... Args>
constexpr optional<T> make_optional(std::initializer_list<U> il,
                                    Args&&... args) {
  return optional<T>(in_place_t(), il,
                     internal_optional::forward<Args>(args)...);
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

}  // namespace gtl
}  // namespace tensorflow

namespace std {

// Normally std::hash specializations are not recommended in tensorflow code,
// but we allow this as it is following a standard library component.
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
