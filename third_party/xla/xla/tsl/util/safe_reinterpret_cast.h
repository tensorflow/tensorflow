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

// IsSafeCast<From, To>::value is true if it is safe to reinterpret_cast a
// value of type From to a value of type To.
//
// This is a subset of the types that are safe to cast, but it's the only
// subset that we need for now. We can add more as needed.
template <typename From, typename To>
struct IsSafeCast : std::false_type {};

// It's safe to cast a type to itself.
template <typename T>
struct IsSafeCast<T, T> : std::true_type {};

// It's safe to cast a pointer to any character pointer.
template <typename From>
struct IsSafeCast<From*, char*> : std::true_type {};
template <typename From>
struct IsSafeCast<From*, std::byte*> : std::true_type {};
template <typename From>
struct IsSafeCast<From*, unsigned char*> : std::true_type {};
template <typename From>
struct IsSafeCast<From*, const char*> : std::true_type {};
template <typename From>
struct IsSafeCast<From*, const std::byte*> : std::true_type {};
template <typename From>
struct IsSafeCast<From*, const unsigned char*> : std::true_type {};

// It's safe to cast a character pointer to a pointer to any type.
template <typename To>
struct IsSafeCast<char*, To*> : std::true_type {};
template <typename To>
struct IsSafeCast<std::byte*, To*> : std::true_type {};
template <typename To>
struct IsSafeCast<unsigned char*, To*> : std::true_type {};
template <typename To>
struct IsSafeCast<const char*, To*> : std::true_type {};
template <typename To>
struct IsSafeCast<const std::byte*, To*> : std::true_type {};
template <typename To>
struct IsSafeCast<const unsigned char*, To*> : std::true_type {};

// It's safe to cast a pointer to/from std::uintptr_t.
template <typename From>
struct IsSafeCast<From*, std::uintptr_t> : std::true_type {};
template <typename To>
struct IsSafeCast<std::uintptr_t, To*> : std::true_type {};

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
