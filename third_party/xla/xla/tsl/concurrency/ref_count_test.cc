/* Copyright 2025 Google LLC. All Rights Reserved.

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

#include "xla/tsl/concurrency/ref_count.h"

#include <cstdint>
#include <type_traits>
#include <utility>

#include "absl/base/nullability.h"
#include "xla/tsl/platform/test.h"

namespace tsl {
namespace {

template <typename T, typename = void>
struct HasAddRef : std::false_type {};

template <typename T>
struct HasAddRef<T, std::void_t<decltype(std::declval<T&>().AddRef())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasDropRef : std::false_type {};

template <typename T>
struct HasDropRef<T, std::void_t<decltype(std::declval<T&>().DropRef())>>
    : std::true_type {};

struct Value : public ReferenceCounted<Value> {
  explicit Value(int32_t value) : value(value) {}
  int32_t value;
};

struct ConstValue : public ReferenceCounted<ConstValue> {
  explicit ConstValue(int32_t value, bool* destroyed = nullptr)
      : value(value), destroyed(destroyed) {}
  ~ConstValue() {
    if (destroyed != nullptr) {
      *destroyed = true;
    }
  }
  int32_t value;
  bool* destroyed;
};

class CustomDestroyConstValue
    : public ReferenceCounted<CustomDestroyConstValue> {
 public:
  explicit CustomDestroyConstValue(bool* custom_destroyed)
      : custom_destroyed_(custom_destroyed) {}

 private:
  friend class ReferenceCounted<CustomDestroyConstValue>;

  void Destroy() const {
    *custom_destroyed_ = true;
    ReferenceCounted::Destroy();
  }

  bool* custom_destroyed_;
};

class NonConstCustomDestroyValue
    : public ReferenceCounted<NonConstCustomDestroyValue> {
 public:
  explicit NonConstCustomDestroyValue(bool* custom_destroyed)
      : custom_destroyed_(custom_destroyed) {}

 private:
  friend class ReferenceCounted<NonConstCustomDestroyValue>;

  void Destroy() {
    *custom_destroyed_ = true;
    ReferenceCounted::Destroy();
  }

  bool* custom_destroyed_;
};

static_assert(HasAddRef<Value>::value);
static_assert(HasAddRef<const Value>::value);
static_assert(HasDropRef<Value>::value);
static_assert(HasDropRef<const Value>::value);

static_assert(HasAddRef<ConstValue>::value);
static_assert(HasAddRef<const ConstValue>::value);
static_assert(HasDropRef<ConstValue>::value);
static_assert(HasDropRef<const ConstValue>::value);
static_assert(std::is_convertible_v<RCReference<ConstValue>,
                                    RCReference<const ConstValue>>);
static_assert(!std::is_convertible_v<RCReference<const ConstValue>,
                                     RCReference<ConstValue>>);

TEST(RefCountTest, NullableRef) {
  absl_nullable RCReference<Value> ref;
  EXPECT_EQ(ref, nullptr);
}

TEST(RefCountTest, NonnullRef) {
  absl_nonnull auto ref = MakeRef<Value>(123);
  EXPECT_EQ(1, ref->NumRef());
  EXPECT_TRUE(ref->IsUnique());
  EXPECT_EQ(123, ref->value);
}

TEST(RefCountTest, ConstRefCountedAddAndDropRef) {
  bool destroyed = false;
  const ConstValue* ptr = new ConstValue(42, &destroyed);
  EXPECT_EQ(ptr->NumRef(), 1);
  EXPECT_TRUE(ptr->IsUnique());

  ptr->AddRef();
  EXPECT_EQ(ptr->NumRef(), 2);
  EXPECT_FALSE(ptr->IsUnique());

  ptr->DropRef();
  EXPECT_FALSE(destroyed);
  EXPECT_EQ(ptr->NumRef(), 1);
  EXPECT_TRUE(ptr->IsUnique());

  ptr->DropRef();
  EXPECT_TRUE(destroyed);
}

TEST(RefCountTest, ConstRefCountedWithRCReference) {
  bool destroyed = false;
  {
    RCReference<ConstValue> mut_ref = MakeRef<ConstValue>(123, &destroyed);
    mut_ref->value = 456;
    RCReference<const ConstValue> const_ref = mut_ref;
    EXPECT_EQ(mut_ref, const_ref);
    EXPECT_EQ(const_ref, mut_ref);
    EXPECT_EQ(const_ref->NumRef(), 2);
    EXPECT_FALSE(const_ref->IsUnique());
    EXPECT_EQ(const_ref->value, 456);

    mut_ref.reset();
    EXPECT_NE(mut_ref, const_ref);
    EXPECT_NE(const_ref, mut_ref);
    EXPECT_FALSE(destroyed);
    EXPECT_TRUE(const_ref->IsUnique());
  }
  EXPECT_TRUE(destroyed);
}

TEST(RefCountTest, ConstRefCountedCustomDestroy) {
  bool custom_destroyed = false;
  {
    RCReference<const CustomDestroyConstValue> ref =
        MakeRef<const CustomDestroyConstValue>(&custom_destroyed);
    RCReference<const CustomDestroyConstValue> ref2 = FormRef(ref.get());
    EXPECT_EQ(ref->NumRef(), 2);
    EXPECT_FALSE(custom_destroyed);
  }
  EXPECT_TRUE(custom_destroyed);
}

TEST(RefCountTest, ConstRefNonConstCustomDestroy) {
  bool custom_destroyed = false;
  {
    RCReference<const NonConstCustomDestroyValue> ref =
        MakeRef<NonConstCustomDestroyValue>(&custom_destroyed);
    RCReference<const NonConstCustomDestroyValue> ref2 = ref;
    EXPECT_EQ(ref->NumRef(), 2);
    EXPECT_FALSE(custom_destroyed);
  }
  EXPECT_TRUE(custom_destroyed);
}

}  // namespace
}  // namespace tsl
