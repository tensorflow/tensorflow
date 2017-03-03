/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/lib/gtl/manual_constructor.h"

#include <stdint.h>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

static int constructor_count_ = 0;

template <int kSize>
struct TestN {
  TestN() { ++constructor_count_; }
  ~TestN() { --constructor_count_; }
  char a[kSize];
};

typedef TestN<1> Test1;
typedef TestN<2> Test2;
typedef TestN<3> Test3;
typedef TestN<4> Test4;
typedef TestN<5> Test5;
typedef TestN<9> Test9;
typedef TestN<15> Test15;

}  // namespace

namespace {

TEST(ManualConstructorTest, Sizeof) {
  CHECK_EQ(sizeof(ManualConstructor<Test1>), sizeof(Test1));
  CHECK_EQ(sizeof(ManualConstructor<Test2>), sizeof(Test2));
  CHECK_EQ(sizeof(ManualConstructor<Test3>), sizeof(Test3));
  CHECK_EQ(sizeof(ManualConstructor<Test4>), sizeof(Test4));
  CHECK_EQ(sizeof(ManualConstructor<Test5>), sizeof(Test5));
  CHECK_EQ(sizeof(ManualConstructor<Test9>), sizeof(Test9));
  CHECK_EQ(sizeof(ManualConstructor<Test15>), sizeof(Test15));

  CHECK_EQ(constructor_count_, 0);
  ManualConstructor<Test1> mt[4];
  CHECK_EQ(sizeof(mt), 4);
  CHECK_EQ(constructor_count_, 0);
  mt[0].Init();
  CHECK_EQ(constructor_count_, 1);
  mt[0].Destroy();
}

TEST(ManualConstructorTest, Alignment) {
  // We want to make sure that ManualConstructor aligns its memory properly
  // on a word barrier.  Otherwise, it might be unexpectedly slow, since
  // memory access will be unaligned.

  struct {
    char a;
    ManualConstructor<void*> b;
  } test1;
  struct {
    char a;
    void* b;
  } control1;

  // TODO(bww): Make these tests more direct with C++11 alignment_of<T>::value.
  EXPECT_EQ(reinterpret_cast<char*>(test1.b.get()) - &test1.a,
            reinterpret_cast<char*>(&control1.b) - &control1.a);
  EXPECT_EQ(reinterpret_cast<intptr_t>(test1.b.get()) % sizeof(control1.b), 0);

  struct {
    char a;
    ManualConstructor<long double> b;
  } test2;
  struct {
    char a;
    long double b;
  } control2;

  EXPECT_EQ(reinterpret_cast<char*>(test2.b.get()) - &test2.a,
            reinterpret_cast<char*>(&control2.b) - &control2.a);
#ifdef ARCH_K8
  EXPECT_EQ(reinterpret_cast<intptr_t>(test2.b.get()) % 16, 0);
#endif
#ifdef ARCH_PIII
  EXPECT_EQ(reinterpret_cast<intptr_t>(test2.b.get()) % 4, 0);
#endif
}

TEST(ManualConstructorTest, DefaultInitialize) {
  struct X {
    X() : x(123) {}
    int x;
  };
  union {
    ManualConstructor<X> x;
    ManualConstructor<int> y;
  } u;
  *u.y = -1;
  u.x.Init();  // should default-initialize u.x
  EXPECT_EQ(123, u.x->x);
}

TEST(ManualConstructorTest, ZeroInitializePOD) {
  union {
    ManualConstructor<int> x;
    ManualConstructor<int> y;
  } u;
  *u.y = -1;
  u.x.Init();  // should not zero-initialize u.x
  EXPECT_EQ(-1, *u.y);
}

}  // namespace
}  // namespace tensorflow
