// Copyright 2025 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

// This file provides a safe_reinterpret_cast function template that is like
// reinterpret_cast, but compiles only if the cast is safe.
//
// In general, reinterpret_cast is unsafe because it can easily cause undefined
// behavior. For example,
//
//     Foo* foo = ...;
//     Bar* bar = reinterpret_cast<Bar*>(foo);
//     *bar = ...;
//
// is undefined behavior unless Foo or Bar is a character type. See
// https://en.cppreference.com/w/cpp/language/reinterpret_cast for more details.
//
// safe_reinterpret_cast is a subset of the casts that are always safe. We can
// add more as needed.

#ifndef XLA_TSL_UTIL_SAFE_REINTERPRET_CAST_H_
#define XLA_TSL_UTIL_SAFE_REINTERPRET_CAST_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tsl {

namespace internal {

// IsByteLike<T>::value is true if T is a byte-like type (char, unsigned char,
// or std::byte).
template <typename T>
struct IsByteLike : std::false_type {};
template <>
struct IsByteLike<char> : std::true_type {};
template <>
struct IsByteLike<unsigned char> : std::true_type {};
template <>
struct IsByteLike<std::byte> : std::true_type {};

// IsCvByteLike<T>::value is true if T is a possibly CV-qualified byte-like type
// (char, unsigned char, or std::byte).
template <typename T>
struct IsCvByteLike : IsByteLike<T> {};
template <typename T>
struct IsCvByteLike<const T> : IsByteLike<T> {};
template <typename T>
struct IsCvByteLike<volatile T> : IsByteLike<T> {};
template <typename T>
struct IsCvByteLike<const volatile T> : IsByteLike<T> {};

// IsSafeCast<From, To>::value is true if it is safe to reinterpret_cast a
// value of type From to a value of type To.
//
// This is a subset of the types that are safe to cast, but it's the only
// subset that we need for now. We can add more as needed.
template <typename From, typename To>
struct IsSafeCast : std::false_type {};

// It's safe to cast a pointer to/from a byte-like type, or to/from the same
// type.
template <typename From, typename To>
struct IsSafeCast<From*, To*>
    : std::integral_constant<bool, IsCvByteLike<From>::value ||
                                       IsCvByteLike<To>::value ||
                                       std::is_same_v<From, To>> {};

// It's safe to cast a pointer to/from std::uintptr_t.
template <typename From>
struct IsSafeCast<From*, std::uintptr_t> : std::true_type {};
template <typename To>
struct IsSafeCast<std::uintptr_t, To*> : std::true_type {};

// It's safe to cast a pointer to/from std::intptr_t.
template <typename From>
struct IsSafeCast<From*, std::intptr_t> : std::true_type {};
template <typename To>
struct IsSafeCast<std::intptr_t, To*> : std::true_type {};

}  // namespace internal

// Like reinterpret_cast, but compiles only if it's safe.
template <typename To, typename From,
          typename = std::enable_if_t<internal::IsSafeCast<From, To>::value>>
To safe_reinterpret_cast(From from) {
  return reinterpret_cast<To>(from);  // REINTERPRET_CAST_OK=for implementing
                                      // safe_reinterpret_cast.
}

}  // namespace tsl

#endif  // XLA_TSL_UTIL_SAFE_REINTERPRET_CAST_H_
