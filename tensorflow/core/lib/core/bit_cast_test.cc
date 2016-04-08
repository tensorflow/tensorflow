/* Copyright 2015 Google Inc. All Rights Reserved.

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

// Unit test for bit_cast template.

#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// Marshall and unmarshall.
// ISO spec C++ section 3.9 promises this will work.

template <int N>
struct marshall {
  char buf[N];
};

template <class T>
void TestMarshall(const T values[], int num_values) {
  for (int i = 0; i < num_values; ++i) {
    T t0 = values[i];
    marshall<sizeof(T)> m0 = bit_cast<marshall<sizeof(T)> >(t0);
    T t1 = bit_cast<T>(m0);
    marshall<sizeof(T)> m1 = bit_cast<marshall<sizeof(T)> >(t1);
    ASSERT_EQ(0, memcmp(&t0, &t1, sizeof(T)));
    ASSERT_EQ(0, memcmp(&m0, &m1, sizeof(T)));
  }
}

// Convert back and forth to an integral type.  The C++ standard does
// not guarantee this will work.
//
// There are implicit assumptions about sizeof(float) and
// sizeof(double). These assumptions are quite extant everywhere.

template <class T, class I>
void TestIntegral(const T values[], int num_values) {
  for (int i = 0; i < num_values; ++i) {
    T t0 = values[i];
    I i0 = bit_cast<I>(t0);
    T t1 = bit_cast<T>(i0);
    I i1 = bit_cast<I>(t1);
    ASSERT_EQ(0, memcmp(&t0, &t1, sizeof(T)));
    ASSERT_EQ(i0, i1);
  }
}

TEST(BitCast, Bool) {
  LOG(INFO) << "Test bool";
  static const bool bool_list[] = {false, true};
  TestMarshall<bool>(bool_list, TF_ARRAYSIZE(bool_list));
}

TEST(BitCast, Int32) {
  static const int32 int_list[] = {0,  1,    100,         2147483647,
                                   -1, -100, -2147483647, -2147483647 - 1};
  TestMarshall<int32>(int_list, TF_ARRAYSIZE(int_list));
}

TEST(BitCast, Int64) {
  static const int64 int64_list[] = {0, 1, 1LL << 40, -1, -(1LL << 40)};
  TestMarshall<int64>(int64_list, TF_ARRAYSIZE(int64_list));
}

TEST(BitCast, Uint64) {
  static const uint64 uint64_list[] = {0, 1, 1LLU << 40, 1LLU << 63};
  TestMarshall<uint64>(uint64_list, TF_ARRAYSIZE(uint64_list));
}

TEST(BitCast, Float) {
  static const float float_list[] = {0.0,  1.0,   -1.0,  10.0,    -10.0,  1e10,
                                     1e20, 1e-10, 1e-20, 2.71828, 3.14159};
  TestMarshall<float>(float_list, TF_ARRAYSIZE(float_list));
  TestIntegral<float, int32>(float_list, TF_ARRAYSIZE(float_list));
  TestIntegral<float, uint32>(float_list, TF_ARRAYSIZE(float_list));
}

TEST(BitCast, Double) {
  static const double double_list[] = {
      0.0,
      1.0,
      -1.0,
      10.0,
      -10.0,
      1e10,
      1e100,
      1e-10,
      1e-100,
      2.718281828459045,
      3.141592653589793238462643383279502884197169399375105820974944};
  TestMarshall<double>(double_list, TF_ARRAYSIZE(double_list));
  TestIntegral<double, int64>(double_list, TF_ARRAYSIZE(double_list));
  TestIntegral<double, uint64>(double_list, TF_ARRAYSIZE(double_list));
}

}  // namespace tensorflow
