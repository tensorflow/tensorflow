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

#include "ral/ral_registry.h"

#include <sys/types.h>

#ifndef PLATFORM_GOOGLE
#include <gtest/gtest.h>
#else
#include "testing/base/public/gunit.h"
#endif

namespace mlir {
namespace disc_ral {

const float kTolerance = 1e-5;

void testVoidReturn(int32_t) {}

void testMemRefVoidReturn(void* data, int x, MemRefType<float, 2> memref) {
  EXPECT_EQ(memref.basePtr, static_cast<float*>(data));
  EXPECT_EQ(memref.data, static_cast<float*>(data) + 1);
  EXPECT_EQ(memref.offset, 0);
  EXPECT_EQ(memref.sizes[0], x);
  EXPECT_EQ(memref.sizes[0], x);
  EXPECT_EQ(memref.strides[0], x);
  EXPECT_EQ(memref.strides[1], 1);
}

bool testI1Return(int32_t x) { return x; }
int16_t testI16Return(int32_t x) { return x; }
int32_t testI32Return(int32_t x) { return x; }
int64_t testI64Return(int32_t x) { return static_cast<uint64_t>(x) << 16; }
float testf32Return(int32_t x) { return 1.0 / x; }
double testf64Return(int32_t x) { return 1.0 / x; }

MemRefType<float, 2> testRank2MemRefReturn(void* data, int idx) {
  MemRefType<float, 2> memref;
  memref.basePtr = static_cast<float*>(data);
  memref.data = static_cast<float*>(data) + 1;
  memref.offset = 0;
  memref.sizes[0] = idx;
  memref.sizes[1] = idx;
  memref.strides[0] = idx;
  memref.strides[1] = 1;
  return memref;
}

MemRefType<float, 0> testRank0MemRefReturn(void* data, int idx) {
  MemRefType<float, 0> memref;
  memref.basePtr = static_cast<float*>(data);
  memref.data = static_cast<float*>(data) + 1;
  memref.offset = 0;
  return memref;
}

TEST(TypeEncoderTest, BasicType) {
  EXPECT_EQ(TypeEncoder<bool>::Invoke(), "i1");
  EXPECT_EQ(TypeEncoder<int16_t>::Invoke(), "i16");
  EXPECT_EQ(TypeEncoder<int32_t>::Invoke(), "i32");
  EXPECT_EQ(TypeEncoder<int64_t>::Invoke(), "i64");
  EXPECT_EQ(TypeEncoder<float>::Invoke(), "f32");
  EXPECT_EQ(TypeEncoder<double>::Invoke(), "f64");
  EXPECT_EQ(TypeEncoder<void*>::Invoke(), "pvoid");
  EXPECT_EQ(TypeEncoder<const void*>::Invoke(), "pvoid");
  EXPECT_EQ(TypeEncoder<void**>::Invoke(), "ppvoid");
  EXPECT_EQ(TypeEncoder<char*>::Invoke(), "pvoid");
  EXPECT_EQ(TypeEncoder<const char*>::Invoke(), "pvoid");
  EXPECT_EQ(TypeEncoder<void>::Invoke(), "void");
}

TEST(FunctionEncoderTest, VoidReturn) {
  EXPECT_EQ(FunctionEncoder<decltype(&testVoidReturn)>::Invoke(""),
            "___i32___void");
  EXPECT_EQ(FunctionEncoder<decltype(&testVoidReturn)>::Invoke("prefix"),
            "prefix___i32___void");
  EXPECT_EQ(FunctionEncoder<decltype(&testMemRefVoidReturn)>::Invoke(""),
            "___pvoid_i32_m2df32___void");
}

TEST(FunctionEncoderTest, PODReturn) {
  EXPECT_EQ(FunctionEncoder<decltype(&testI1Return)>::Invoke(""),
            "___i32___i1");
  EXPECT_EQ(FunctionEncoder<decltype(&testI16Return)>::Invoke(""),
            "___i32___i16");
  EXPECT_EQ(FunctionEncoder<decltype(&testI32Return)>::Invoke(""),
            "___i32___i32");
  EXPECT_EQ(FunctionEncoder<decltype(&testI64Return)>::Invoke(""),
            "___i32___i64");
  EXPECT_EQ(FunctionEncoder<decltype(&testf32Return)>::Invoke(""),
            "___i32___f32");
  EXPECT_EQ(FunctionEncoder<decltype(&testf64Return)>::Invoke(""),
            "___i32___f64");
}

TEST(FunctionEncoderTest, MemRefReturn) {
  EXPECT_EQ(FunctionEncoder<decltype(&testRank2MemRefReturn)>::Invoke(""),
            "___pvoid_i32___m2df32");
  EXPECT_EQ(FunctionEncoder<decltype(&testRank0MemRefReturn)>::Invoke(""),
            "___pvoid_i32___m0df32");
}

TEST(FunctionWrapperTest, PODTest) {
  bool ret_i1;
  int16_t ret_i16;
  int32_t ret_i32;
  int64_t ret_i64;
  float ret_f32;
  double ret_f64;

  ral_func_t func_i1 =
      FunctionWrapper<decltype(&testI1Return), &testI1Return>::Invoke;
  ral_func_t func_i16 =
      FunctionWrapper<decltype(&testI16Return), &testI16Return>::Invoke;
  ral_func_t func_i32 =
      FunctionWrapper<decltype(&testI32Return), &testI32Return>::Invoke;
  ral_func_t func_i64 =
      FunctionWrapper<decltype(&testI64Return), &testI64Return>::Invoke;
  ral_func_t func_f32 =
      FunctionWrapper<decltype(&testf32Return), &testf32Return>::Invoke;
  ral_func_t func_f64 =
      FunctionWrapper<decltype(&testf64Return), &testf64Return>::Invoke;

  auto apply = [&](ral_func_t func, int32_t x, void* out) {
    void* args[] = {&x, out};
    func(args);
  };

  // test bool return
  ret_i1 = true;
  apply(func_i1, 0, &ret_i1);
  EXPECT_EQ(ret_i1, false);
  apply(func_i1, 1, &ret_i1);
  EXPECT_EQ(ret_i1, true);

  // test int16_t return
  ret_i16 = 0;
  apply(func_i16, -1, &ret_i16);
  EXPECT_EQ(ret_i16, -1);
  apply(func_i16, 3, &ret_i16);
  EXPECT_EQ(ret_i16, 3);

  // test int32_t return
  ret_i32 = 0;
  apply(func_i32, ((unsigned)(-1)) << 17, &ret_i32);  // NOLINT
  EXPECT_EQ(ret_i32, ((unsigned)(-1)) << 17);
  apply(func_i32, 3 << 17, &ret_i32);
  EXPECT_EQ(ret_i32, 3 << 17);

  // test int64_t return
  ret_i64 = 0;
  apply(func_i64, -1, &ret_i64);
  EXPECT_EQ(ret_i64, ((uint64_t)(-1)) << 16);
  apply(func_i64, 3, &ret_i64);
  EXPECT_EQ(ret_i64, ((uint64_t)3) << 16);

  // test float return
  ret_f32 = 0;
  apply(func_f32, -3, &ret_f32);
  EXPECT_NEAR(ret_f32, -0.333333, kTolerance);
  apply(func_f32, 3, &ret_f32);
  EXPECT_NEAR(ret_f32, 0.333333, kTolerance);

  // test double return
  ret_f64 = 0;
  apply(func_f64, -3, &ret_f64);
  EXPECT_NEAR(ret_f64, -0.333333, kTolerance);
  apply(func_f64, 3, &ret_f64);
  EXPECT_NEAR(ret_f64, 0.333333, kTolerance);
}

TEST(FunctionWrapperTest, MemRefTest) {
  float v;
  MemRefType<float, 0> m0d;
  MemRefType<float, 2> m2d;

  ral_func_t func_m0d = FunctionWrapper<decltype(&testRank0MemRefReturn),
                                        &testRank0MemRefReturn>::Invoke;
  ral_func_t func_m2d = FunctionWrapper<decltype(&testRank2MemRefReturn),
                                        &testRank2MemRefReturn>::Invoke;
  ral_func_t func_verify_m2d = FunctionWrapper<decltype(&testMemRefVoidReturn),
                                               &testMemRefVoidReturn>::Invoke;

  auto applySetter = [&](ral_func_t func, float* data, int32_t x, void* out) {
    void* args[] = {&data, &x, out};
    func(args);
  };
  auto applyVerifier = [&](ral_func_t func, float* data, int32_t x,
                           MemRefType<float, 2> out) {
    void* args[] = {&data,          &x,
                    &out.basePtr,   &out.data,
                    &out.offset,    &out.sizes[0],
                    &out.sizes[1],  &out.strides[0],
                    &out.strides[1]};
    func(args);
  };

  applySetter(func_m0d, &v, 2, &m0d);
  EXPECT_EQ(m0d.basePtr, &v);
  EXPECT_EQ(m0d.data, &v + 1);
  EXPECT_EQ(m0d.offset, 0);

  applySetter(func_m2d, &v, 3, &m2d);
  applyVerifier(func_verify_m2d, &v, 3, m2d);
}

TEST(FunctionRegistry, Test) {
  ral_func_t f1 =
      FunctionWrapper<decltype(&testI1Return), &testI1Return>::Invoke;
  EXPECT_TRUE(FunctionRegistry::Global().Register("f1", f1));
  EXPECT_NE(FunctionRegistry::Global().Find("f1"), nullptr);
  EXPECT_EQ(FunctionRegistry::Global().Find("f2"), nullptr);
}

}  // namespace disc_ral
}  // namespace mlir
