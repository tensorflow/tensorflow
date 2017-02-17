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

#include "tensorflow/core/lib/core/refcount.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace core {
namespace {

static int constructed = 0;
static int destroyed = 0;

class MyRef : public RefCounted {
 public:
  MyRef() { constructed++; }
  ~MyRef() override { destroyed++; }
};

class RefTest : public ::testing::Test {
 public:
  RefTest() {
    constructed = 0;
    destroyed = 0;
  }
};

TEST_F(RefTest, New) {
  MyRef* ref = new MyRef;
  ASSERT_EQ(1, constructed);
  ASSERT_EQ(0, destroyed);
  ref->Unref();
  ASSERT_EQ(1, constructed);
  ASSERT_EQ(1, destroyed);
}

TEST_F(RefTest, RefUnref) {
  MyRef* ref = new MyRef;
  ASSERT_EQ(1, constructed);
  ASSERT_EQ(0, destroyed);
  ref->Ref();
  ASSERT_EQ(0, destroyed);
  ref->Unref();
  ASSERT_EQ(0, destroyed);
  ref->Unref();
  ASSERT_EQ(1, destroyed);
}

TEST_F(RefTest, RefCountOne) {
  MyRef* ref = new MyRef;
  ASSERT_TRUE(ref->RefCountIsOne());
  ref->Unref();
}

TEST_F(RefTest, RefCountNotOne) {
  MyRef* ref = new MyRef;
  ref->Ref();
  ASSERT_FALSE(ref->RefCountIsOne());
  ref->Unref();
  ref->Unref();
}

TEST_F(RefTest, ConstRefUnref) {
  const MyRef* cref = new MyRef;
  ASSERT_EQ(1, constructed);
  ASSERT_EQ(0, destroyed);
  cref->Ref();
  ASSERT_EQ(0, destroyed);
  cref->Unref();
  ASSERT_EQ(0, destroyed);
  cref->Unref();
  ASSERT_EQ(1, destroyed);
}

TEST_F(RefTest, ReturnOfUnref) {
  MyRef* ref = new MyRef;
  ref->Ref();
  EXPECT_FALSE(ref->Unref());
  EXPECT_TRUE(ref->Unref());
}

TEST_F(RefTest, ScopedUnref) {
  { ScopedUnref unref(new MyRef); }
  EXPECT_EQ(destroyed, 1);
}

TEST_F(RefTest, ScopedUnref_Nullptr) {
  { ScopedUnref unref(nullptr); }
  EXPECT_EQ(destroyed, 0);
}

}  // namespace
}  // namespace core
}  // namespace tensorflow
