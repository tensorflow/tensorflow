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

#include "tensorflow/tsl/platform/refcount.h"

#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/test.h"
#include "tensorflow/tsl/platform/threadpool.h"

namespace tsl {
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
  WeakPtr<ObjType> weakptr(obj);

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
    WeakPtr<ObjType> weakptr(obj);

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
  if (hit_destructed == 0) {
    LOG(WARNING) << "The destructed weakref test branch is not exercised.";
  }
  if (hit_destructed == 200) {
    LOG(WARNING) << "The valid weakref test branch is not exercised.";
  }
}

TEST(WeakPtr, NotifyCalled) {
  auto obj = new ObjType();
  int num_calls1 = 0;
  int num_calls2 = 0;

  auto notify_fn1 = [&num_calls1]() { num_calls1++; };
  auto notify_fn2 = [&num_calls2]() { num_calls2++; };
  WeakPtr<ObjType> weakptr1(obj, notify_fn1);
  WeakPtr<ObjType> weakptr2(obj, notify_fn2);

  ASSERT_TRUE(obj->RefCountIsOne());
  EXPECT_EQ(obj->WeakRefCount(), 2);
  EXPECT_NE(weakptr1.GetNewRef(), nullptr);
  EXPECT_NE(weakptr2.GetNewRef(), nullptr);

  EXPECT_EQ(num_calls1, 0);
  EXPECT_EQ(num_calls2, 0);
  obj->Unref();
  EXPECT_EQ(weakptr1.GetNewRef(), nullptr);
  EXPECT_EQ(weakptr2.GetNewRef(), nullptr);
  EXPECT_EQ(num_calls1, 1);
  EXPECT_EQ(num_calls2, 1);
}

TEST(WeakPtr, CopyTargetCalled) {
  auto obj = new ObjType();
  int num_calls1 = 0;
  int num_calls2 = 0;

  auto notify_fn1 = [&num_calls1]() { num_calls1++; };
  auto notify_fn2 = [&num_calls2]() { num_calls2++; };

  WeakPtr<ObjType> weakptr1(obj, notify_fn1);
  WeakPtr<ObjType> weakptr2(obj, notify_fn2);
  WeakPtr<ObjType> weakptr3(weakptr1);

  weakptr2 = weakptr1;

  ASSERT_TRUE(obj->RefCountIsOne());
  EXPECT_EQ(obj->WeakRefCount(), 3);
  EXPECT_NE(weakptr2.GetNewRef(), nullptr);
  EXPECT_NE(weakptr3.GetNewRef(), nullptr);

  EXPECT_EQ(num_calls1, 0);
  EXPECT_EQ(num_calls2, 0);
  obj->Unref();
  EXPECT_EQ(weakptr2.GetNewRef(), nullptr);
  EXPECT_EQ(weakptr3.GetNewRef(), nullptr);
  EXPECT_EQ(num_calls1, 3);
  EXPECT_EQ(num_calls2, 0);
}

TEST(WeakPtr, MoveTargetNotCalled) {
  auto obj = new ObjType();
  int num_calls1 = 0;
  int num_calls2 = 0;
  int num_calls3 = 0;

  auto notify_fn1 = [&num_calls1]() { num_calls1++; };
  auto notify_fn2 = [&num_calls2]() { num_calls2++; };
  auto notify_fn3 = [&num_calls3]() { num_calls3++; };
  WeakPtr<ObjType> weakptr1(obj, notify_fn1);
  WeakPtr<ObjType> weakptr2(obj, notify_fn2);
  WeakPtr<ObjType> weakptr3(WeakPtr<ObjType>(obj, notify_fn3));

  weakptr2 = std::move(weakptr1);

  ASSERT_TRUE(obj->RefCountIsOne());
  EXPECT_EQ(obj->WeakRefCount(), 2);
  EXPECT_NE(weakptr2.GetNewRef(), nullptr);
  EXPECT_NE(weakptr3.GetNewRef(), nullptr);

  EXPECT_EQ(num_calls1, 0);
  EXPECT_EQ(num_calls2, 0);
  EXPECT_EQ(num_calls3, 0);
  obj->Unref();
  EXPECT_EQ(weakptr2.GetNewRef(), nullptr);
  EXPECT_EQ(weakptr3.GetNewRef(), nullptr);
  EXPECT_EQ(num_calls1, 1);
  EXPECT_EQ(num_calls2, 0);
  EXPECT_EQ(num_calls3, 1);
}

TEST(WeakPtr, DestroyedNotifyNotCalled) {
  auto obj = new ObjType();
  int num_calls = 0;
  auto notify_fn = [&num_calls]() { num_calls++; };
  { WeakPtr<ObjType> weakptr(obj, notify_fn); }
  ASSERT_TRUE(obj->RefCountIsOne());
  EXPECT_EQ(obj->WeakRefCount(), 0);

  EXPECT_EQ(num_calls, 0);
  obj->Unref();
  EXPECT_EQ(num_calls, 0);
}

}  // namespace
}  // namespace core
}  // namespace tsl
