/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/core/async/interop/variant.h"

#include <cstddef>
#include <string>

#include <gtest/gtest.h>

namespace tflite::interop {
namespace {

TEST(VariantTest, IntTest) {
  {
    Variant a(1);
    EXPECT_EQ(1, *a.Get<int>());
  }
  {
    Variant a(1);
    a.Set(2);
    EXPECT_EQ(2, *a.Get<int>());
  }
  {
    Variant a(42);
    Variant b(a);
    EXPECT_EQ(42, *b.Get<int>());
  }
  {
    Variant a(42);
    EXPECT_EQ(42, *static_cast<const int*>(a.GetPtr()));
  }
  {
    Variant a(42);
    Variant b(42);
    EXPECT_EQ(a, b);
    b.Set(21);
    EXPECT_NE(a, b);
  }
}

TEST(VariantTest, SizeTTest) {
  {
    size_t v = 1;
    Variant a(v);
    EXPECT_EQ(1, *a.Get<size_t>());
  }
  {
    size_t v = 1;
    Variant a(v);
    size_t t = 2;
    a.Set(t);
    EXPECT_EQ(2, *a.Get<size_t>());
  }
  {
    size_t v = 42;
    Variant a(v);
    Variant b(a);
    EXPECT_EQ(42, *b.Get<size_t>());
  }
  {
    size_t v = 42;
    Variant a(v);
    EXPECT_EQ(42, *static_cast<const size_t*>(a.GetPtr()));
  }
  {
    Variant a(size_t(42));
    Variant b(size_t(42));
    EXPECT_EQ(a, b);
    b.Set(size_t(21));
    EXPECT_NE(a, b);
  }
}

TEST(VariantTest, StringTest) {
  {
    const char v[] = "string";
    Variant a(v);
    EXPECT_EQ(v, *a.Get<const char*>());
  }
  {
    const char v[] = "string";
    Variant a(v);
    const char t[] = "another string";
    a.Set(t);
    EXPECT_EQ(t, *a.Get<const char*>());
  }
  {
    const char v[] = "string";
    Variant a(v);
    Variant b(a);
    EXPECT_EQ(v, *b.Get<const char*>());
  }
  {
    const char v[] = "string";
    Variant a(v);
    EXPECT_EQ(v, *static_cast<const char* const*>(a.GetPtr()));
  }
  {
    const char v[] = "string";
    Variant a(v);
    std::string str = "string";
    Variant b(str.c_str());
    EXPECT_EQ(a, b);
    b.Set("another string");
    EXPECT_NE(a, b);
  }
}

TEST(VariantTest, TypeNotMatch) {
  Variant a(1);
  EXPECT_EQ(nullptr, a.Get<size_t>());
}

}  // namespace
}  // namespace tflite::interop
