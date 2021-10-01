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

#include "tensorflow/core/platform/refcount.h"

#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace core {
namespace {

class RefTest : public ::testing::Test {
 public:
  RefTest() {
    constructed_ = 0;
    destroyed_ = 0;
  }

  static int constructed_;
  static int destroyed_;
};

int RefTest::constructed_;
int RefTest::destroyed_;

class MyRef : public RefCounted {
 public:
  MyRef() { RefTest::constructed_++; }
  ~MyRef() override { RefTest::destroyed_++; }
};

TEST_F(RefTest, New) {
  MyRef* ref = new MyRef;
  ASSERT_EQ(1, constructed_);
  ASSERT_EQ(0, destroyed_);
  ref->Unref();
  ASSERT_EQ(1, constructed_);
  ASSERT_EQ(1, destroyed_);
}

TEST_F(RefTest, RefUnref) {
  MyRef* ref = new MyRef;
  ASSERT_EQ(1, constructed_);
  ASSERT_EQ(0, destroyed_);
  ref->Ref();
  ASSERT_EQ(0, destroyed_);
  ref->Unref();
  ASSERT_EQ(0, destroyed_);
  ref->Unref();
  ASSERT_EQ(1, destroyed_);
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
  ASSERT_EQ(1, constructed_);
  ASSERT_EQ(0, destroyed_);
  cref->Ref();
  ASSERT_EQ(0, destroyed_);
  cref->Unref();
  ASSERT_EQ(0, destroyed_);
  cref->Unref();
  ASSERT_EQ(1, destroyed_);
}

TEST_F(RefTest, ReturnOfUnref) {
  MyRef* ref = new MyRef;
  ref->Ref();
  EXPECT_FALSE(ref->Unref());
  EXPECT_TRUE(ref->Unref());
}

TEST_F(RefTest, ScopedUnref) {
  { ScopedUnref unref(new MyRef); }
  EXPECT_EQ(destroyed_, 1);
}

TEST_F(RefTest, ScopedUnref_Nullptr) {
  { ScopedUnref unref(nullptr); }
  EXPECT_EQ(destroyed_, 0);
}

class ObjType : public WeakRefCounted {};

TEST(WeakPtr, SingleThread) {
  auto obj = new ObjType();
  auto weakptr = WeakPtr<ObjType>(obj);

  ASSERT_TRUE(obj->RefCountIsOne());
  EXPECT_EQ(obj->WeakRefCount(), 1);
  EXPECT_NE(weakptr.GetNewRef(), nullptr);

  obj->Unref();
  EXPECT_EQ(weakptr.GetNewRef(), nullptr);
}

TEST(WeakPtr, MultiThreadedWeakRef) {
  // Exercise 100 times to make sure both branches of fn are hit.
  std::atomic<int> hit_destructed{0};

  auto env = Env::Default();

  for (int i = 0; i < 100; i++) {
    auto obj = new ObjType();
    auto weakptr = WeakPtr<ObjType>(obj);

    bool obj_destructed = false;
    EXPECT_EQ(obj->WeakRefCount(), 1);

    auto fn = [&]() {
      auto ref = weakptr.GetNewRef();
      if (ref != nullptr) {
        EXPECT_EQ(ref.get(), obj);
        EXPECT_EQ(ref->WeakRefCount(), 1);
        EXPECT_GE(ref->RefCount(), 1);
      } else {
        hit_destructed++;
        EXPECT_TRUE(obj_destructed);
      }
    };

    auto t1 = env->StartThread(ThreadOptions{}, "thread-1", fn);
    auto t2 = env->StartThread(ThreadOptions{}, "thread-2", fn);

    env->SleepForMicroseconds(10);
    obj_destructed = true;  // This shall run before weakref is purged.
    obj->Unref();

    delete t1;
    delete t2;

    EXPECT_EQ(weakptr.GetNewRef(), nullptr);
  }
  ASSERT_GT(hit_destructed, 0);
  ASSERT_LT(hit_destructed, 200);  // 2 threads per iterations.
}
}  // namespace
}  // namespace core
}  // namespace tensorflow
