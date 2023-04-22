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

#include "tensorflow/core/platform/intrusive_ptr.h"

#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace core {
namespace {

TEST(IntrusivePtr, ConstructorAddRefFalse) {
  auto ptr = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  // This is needed so that the compiler does not optimize away dead code.
  ASSERT_TRUE(ptr->RefCountIsOne());
  // Test that there is no leak.
}

TEST(IntrusivePtr, ConstructorAddRefTrue) {
  auto raw = new RefCounted();
  auto ptr = IntrusivePtr<RefCounted>(raw, /*add_ref=*/true);
  ASSERT_FALSE(raw->RefCountIsOne());
  raw->Unref();
  ASSERT_TRUE(raw->RefCountIsOne());
}

TEST(IntrusivePtr, CopyConstructor) {
  auto ptr1 = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  auto ptr2 = IntrusivePtr<RefCounted>(ptr1);
  ASSERT_FALSE(ptr2->RefCountIsOne());
}

TEST(IntrusivePtr, CopyAssignment) {
  auto ptr1 = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  auto raw = new RefCounted();
  auto ptr2 = IntrusivePtr<RefCounted>(raw, /*add_ref=*/true);
  ptr2 = ptr1;
  ASSERT_EQ(ptr1.get(), ptr2.get());
  ASSERT_FALSE(ptr2->RefCountIsOne());
  ASSERT_TRUE(raw->RefCountIsOne());
  raw->Unref();
}

TEST(IntrusivePtr, CopyAssignmentIntoEmpty) {
  auto ptr1 = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  auto ptr2 = IntrusivePtr<RefCounted>();
  ptr2 = ptr1;
  ASSERT_FALSE(ptr2->RefCountIsOne());
}

TEST(IntrusivePtr, MoveConstructor) {
  auto ptr1 = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  auto ptr2 = IntrusivePtr<RefCounted>(std::move(ptr1));
  ASSERT_TRUE(ptr2->RefCountIsOne());
  ASSERT_EQ(ptr1.get(), nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST(IntrusivePtr, MoveAssignment) {
  auto ptr1 = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  auto ptr2 = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  ptr2 = std::move(ptr1);
  ASSERT_TRUE(ptr2->RefCountIsOne());
  ASSERT_EQ(ptr1.get(), nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST(IntrusivePtr, MoveAssignmentIntoEmpty) {
  auto ptr1 = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  auto ptr2 = IntrusivePtr<RefCounted>();
  ptr2 = std::move(ptr1);
  ASSERT_TRUE(ptr2->RefCountIsOne());
  ASSERT_EQ(ptr1.get(), nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST(IntrusivePtr, MoveAssignmentAlias) {
  auto ptr = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  auto& ptr_alias = ptr;
  ptr = std::move(ptr_alias);
  ASSERT_TRUE(ptr->RefCountIsOne());
}

TEST(IntrusivePtr, Reset) {
  auto ptr = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  ptr.reset(new RefCounted(), /*add_ref=*/false);
  ASSERT_TRUE(ptr->RefCountIsOne());
  // Test no leak.
}

TEST(IntrusivePtr, ResetIntoEmpty) {
  auto ptr = IntrusivePtr<RefCounted>();
  ptr.reset(new RefCounted(), /*add_ref=*/false);
  ASSERT_TRUE(ptr->RefCountIsOne());
  // Test no leak.
}

TEST(IntrusivePtr, ResetAlias) {
  auto ptr = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  ASSERT_TRUE(ptr->RefCountIsOne());
  ptr.reset(ptr.get(), /*add_ref=*/false);  // No-op.
  ASSERT_TRUE(ptr->RefCountIsOne());
}

TEST(IntrusivePtr, ResetRefBeforeUnref) {
  class Foo : public RefCounted {
   public:
    explicit Foo(char label, Foo* ptr = nullptr)
        : label_(label), ptr_(ptr, false) {}
    char label_;
    IntrusivePtr<Foo> ptr_;
  };
  IntrusivePtr<Foo> x(new Foo{'a', new Foo{'b', new Foo{'c'}}}, false);
  // This test ensures that reset calls Ref on the new handle before unreffing
  // the current handle to avoid subtle use-after-delete bugs.
  // Here if we were to call Unref first, we will Unref the "Foo" with the
  // label 'b', thereby destroying it.  This will in turn Unref 'c' and destroy
  // that. So reset would try to Ref a deleted object. Calling
  // x->ptr_->ptr_.Ref() before x->ptr_.Unref() avoids this.
  x->ptr_ = x->ptr_->ptr_;
}

TEST(IntrusivePtr, ResetStealPtrBeforeUnref) {
  class Foo : public RefCounted {
   public:
    explicit Foo(char label, Foo* ptr = nullptr)
        : label_(label), ptr_(ptr, false) {}
    char label_;
    IntrusivePtr<Foo> ptr_;
  };
  IntrusivePtr<Foo> x(new Foo{'a', new Foo{'b', new Foo{'c'}}}, false);
  // This test ensures that move assignment clears the handle_ of the moved
  // object before Unreffing the current handle_.
  x->ptr_ = std::move(x->ptr_->ptr_);
}

TEST(IntrusivePtr, Detach) {
  auto ptr = IntrusivePtr<RefCounted>(new RefCounted(), /*add_ref=*/false);
  ASSERT_TRUE(ptr->RefCountIsOne());
  auto raw = ptr.detach();
  ASSERT_TRUE(raw->RefCountIsOne());
  raw->Unref();
}
}  // namespace
}  // namespace core
}  // namespace tensorflow
