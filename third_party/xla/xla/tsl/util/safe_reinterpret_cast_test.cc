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

#include "xla/tsl/util/safe_reinterpret_cast.h"

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

namespace tsl {
namespace {

TEST(SafeReinterpretCast, CanCastPointerToFromConstCharPointer) {
  const int x = 42;
  const char* const char_p = safe_reinterpret_cast<const char*>(&x);
  EXPECT_EQ(
      char_p,                              //
      reinterpret_cast<const char*>(&x));  // REINTERPRET_CAST_OK=for testing.

  const int* const int_p = safe_reinterpret_cast<const int*>(char_p);
  EXPECT_EQ(int_p, &x);
}

TEST(SafeReinterpretCast, CanCastPointerToFromConstBytePointer) {
  const int x = 42;
  const std::byte* const char_p = safe_reinterpret_cast<const std::byte*>(&x);
  EXPECT_EQ(
      char_p,                              //
      reinterpret_cast<const std::byte*>(  // REINTERPRET_CAST_OK=for testing.
          &x));

  const int* const int_p = safe_reinterpret_cast<const int*>(char_p);
  EXPECT_EQ(int_p, &x);
}

TEST(SafeReinterpretCast, CanCastPointerToFromConstUnsignedCharPointer) {
  const int x = 42;
  const unsigned char* const char_p =
      safe_reinterpret_cast<const unsigned char*>(&x);
  EXPECT_EQ(char_p,                                  //
            reinterpret_cast<const unsigned char*>(  // REINTERPRET_CAST_OK=for
                                                     // testing.
                &x));

  const int* const int_p = safe_reinterpret_cast<const int*>(char_p);
  EXPECT_EQ(int_p, &x);
}

TEST(SafeReinterpretCast, CanCastPointerToFromMutableCharPointer) {
  int x = 42;
  char* const char_p = safe_reinterpret_cast<char*>(&x);
  EXPECT_EQ(char_p,                        //
            reinterpret_cast<char*>(&x));  // REINTERPRET_CAST_OK=for testing.

  int* const int_p = safe_reinterpret_cast<int*>(char_p);
  EXPECT_EQ(int_p, &x);
}

TEST(SafeReinterpretCast, CanCastBetweenByteLikePointers) {
  char x = 'A';
  std::byte* const byte_p = safe_reinterpret_cast<std::byte*>(&x);
  EXPECT_EQ(byte_p,                        //
            reinterpret_cast<std::byte*>(  // REINTERPRET_CAST_OK=for testing.
                &x));

  unsigned char* const unsigned_char_p =
      safe_reinterpret_cast<unsigned char*>(&x);
  EXPECT_EQ(unsigned_char_p,                   //
            reinterpret_cast<unsigned char*>(  // REINTERPRET_CAST_OK=for
                                               // testing.
                &x));
}

TEST(SafeReinterpretCast, CanCastPointerToFromStdUintptrT) {
  const int x = 42;
  const std::uintptr_t uintptr_t_p = safe_reinterpret_cast<std::uintptr_t>(&x);
  EXPECT_EQ(
      uintptr_t_p,                       //
      reinterpret_cast<std::uintptr_t>(  // REINTERPRET_CAST_OK=for testing.
          &x));
  EXPECT_EQ(safe_reinterpret_cast<const int*>(uintptr_t_p), &x);
}

TEST(SafeReinterpretCast, CanCastPointerToFromStdIntptrT) {
  const int x = 42;
  const std::intptr_t intptr_t_p = safe_reinterpret_cast<std::intptr_t>(&x);
  EXPECT_EQ(
      intptr_t_p,                       //
      reinterpret_cast<std::intptr_t>(  // REINTERPRET_CAST_OK=for testing.
          &x));
  EXPECT_EQ(safe_reinterpret_cast<const int*>(intptr_t_p), &x);
}

TEST(SafeReinterpretCast, CanCastNullptrToStdUintptrT) {
  const std::uintptr_t n = safe_reinterpret_cast<std::uintptr_t>(nullptr);
  EXPECT_EQ(safe_reinterpret_cast<const void*>(n), nullptr);
}

TEST(SafeReinterpretCast, CanCastNullptrToStdIntptrT) {
  const std::intptr_t n = safe_reinterpret_cast<std::intptr_t>(nullptr);
  EXPECT_EQ(safe_reinterpret_cast<const void*>(n), nullptr);
}

TEST(SafeReinterpretCast, CanCastPointerToFromSameType) {
  const int x = 42;
  const int* const int_p = safe_reinterpret_cast<const int*>(&x);
  EXPECT_EQ(int_p, &x);

  char y = 'A';
  char* const char_p = safe_reinterpret_cast<char*>(&y);
  EXPECT_EQ(char_p, &y);
}

TEST(SafeReinterpretCast, CanCastPointerToRestrictPointer) {
  const int x = 42;
  const char* __restrict const char_p =
      safe_reinterpret_cast<const char* __restrict>(&x);
  EXPECT_EQ(char_p,                         //
            reinterpret_cast<const char*>(  // REINTERPRET_CAST_OK=for testing.
                &x));
}

TEST(SafeReinterpretCast, CanCastRestrictPointerToPointer) {
  const int x = 42;
  const int* __restrict const int_p = &x;
  const char* const char_p = safe_reinterpret_cast<const char*>(int_p);
  EXPECT_EQ(char_p,                         //
            reinterpret_cast<const char*>(  // REINTERPRET_CAST_OK=for testing.
                &x));
}

TEST(SafeReinterpretCast, CanCastRestrictPointerToRestrictPointer) {
  const int x = 42;
  const int* __restrict const int_p = &x;
  const char* __restrict const char_p =
      safe_reinterpret_cast<const char* __restrict>(int_p);
  EXPECT_EQ(char_p,                         //
            reinterpret_cast<const char*>(  // REINTERPRET_CAST_OK=for testing.
                &x));
}

void Dummy() {}

TEST(SafeReinterpretCast, CanCastFuncPointerToFromVoidPointer) {
  void* const void_p = safe_reinterpret_cast<void*>(&Dummy);
  void (*func_p)() = safe_reinterpret_cast<void (*)()>(void_p);
  EXPECT_EQ(func_p, &Dummy);
}

TEST(SafeReinterpretCast, CanCastDataPointerToFromVoidPointer) {
  int x = 42;
  void* const void_p = safe_reinterpret_cast<void*>(&x);
  int* const int_p = safe_reinterpret_cast<int*>(void_p);
  EXPECT_EQ(int_p, &x);
}

}  // namespace
}  // namespace tsl
