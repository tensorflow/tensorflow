/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/python/ifrt/ref_wrapper.h"

#include <utility>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {
namespace {

class BaseClass : public tsl::ReferenceCounted<BaseClass> {
 public:
  BaseClass() = default;
  virtual ~BaseClass() = default;

  virtual absl::string_view name() const = 0;

  bool operator==(const BaseClass& other) const {
    return name() == other.name();
  }
  bool operator!=(const BaseClass& other) const { return !(*this == other); }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const BaseClass& obj) {
    sink.Append(obj.name());
  }

  template <typename H>
  friend H AbslHashValue(H h, const BaseClass& obj) {
    return H::combine(std::move(h), obj.name());
  }
};

class DerivedClass : public BaseClass {
 public:
  explicit DerivedClass(absl::string_view name) : name_(name) {}
  ~DerivedClass() override = default;

  absl::string_view name() const override { return name_; }

 private:
  absl::string_view name_;
};

using BaseClassRef = ::xla::ifrt::RCReferenceWrapper<BaseClass>;
using DerivedClassRef = ::xla::ifrt::RCReferenceWrapper<DerivedClass>;

TEST(RCReferenceWrapperTest, DefaultConstructor) { DerivedClassRef ref; }

TEST(RCReferenceWrapperTest, Constructor) {
  DerivedClassRef ref(tsl::MakeRef<DerivedClass>("test"));
}

TEST(RCReferenceWrapperTest, Dereference) {
  DerivedClassRef ref(tsl::MakeRef<DerivedClass>("test"));
  EXPECT_EQ(ref->name(), "test");
  EXPECT_EQ(ref.get()->name(), "test");
  EXPECT_EQ((*ref).name(), "test");
}

TEST(RCReferenceWrapperTest, ConstructorWithSameType) {
  DerivedClassRef ref1(tsl::MakeRef<DerivedClass>("test"));
  DerivedClassRef ref2(ref1);
  EXPECT_EQ(ref1->name(), "test");
  EXPECT_EQ(ref2->name(), "test");
  DerivedClassRef ref3(std::move(ref2));
  EXPECT_EQ(ref3->name(), "test");
}

TEST(RCReferenceWrapperTest, ConstructorWithBaseType) {
  BaseClassRef ref1(tsl::MakeRef<DerivedClass>("test"));
  BaseClassRef ref2(ref1);
  EXPECT_EQ(ref1->name(), "test");
  EXPECT_EQ(ref2->name(), "test");
  BaseClassRef ref3(std::move(ref2));
  EXPECT_EQ(ref3->name(), "test");
}

TEST(RCReferenceWrapperTest, CastToBool) {
  DerivedClassRef ref(tsl::MakeRef<DerivedClass>("test"));
  EXPECT_TRUE(!!ref);
  ref.reset();
  EXPECT_FALSE(!!ref);
}

TEST(RCReferenceWrapperTest, AssignmentWithSameType) {
  DerivedClassRef ref1(tsl::MakeRef<DerivedClass>("test"));
  DerivedClassRef ref2;
  ref2 = ref1;
  EXPECT_EQ(ref1->name(), "test");
  EXPECT_EQ(ref2->name(), "test");
  DerivedClassRef ref3;
  ref3 = std::move(ref2);
  EXPECT_EQ(ref3->name(), "test");
}

TEST(RCReferenceWrapperTest, AssignmentWithBaseType) {
  BaseClassRef ref1(tsl::MakeRef<DerivedClass>("test"));
  BaseClassRef ref2;
  ref2 = ref1;
  EXPECT_EQ(ref1->name(), "test");
  EXPECT_EQ(ref2->name(), "test");
  BaseClassRef ref3;
  ref3 = std::move(ref2);
  EXPECT_EQ(ref3->name(), "test");
}

TEST(RCReferenceWrapperTest, ComparisonWithNonnull) {
  DerivedClassRef ref1(tsl::MakeRef<DerivedClass>("test"));
  DerivedClassRef ref2(tsl::MakeRef<DerivedClass>("test"));
  DerivedClassRef ref3(tsl::MakeRef<DerivedClass>("test3"));
  EXPECT_EQ(ref1, ref1);
  EXPECT_EQ(ref1, ref2);
  EXPECT_NE(ref1, ref3);
}

TEST(RCReferenceWrapperTest, ComparisonWithNullptr) {
  DerivedClassRef ref1(tsl::MakeRef<DerivedClass>("test"));
  DerivedClassRef ref2;
  EXPECT_NE(ref1, nullptr);
  EXPECT_NE(nullptr, ref1);
  EXPECT_EQ(ref2, nullptr);
  EXPECT_EQ(nullptr, ref2);
}

TEST(RCReferenceWrapperTest, ComparisonWithWrappedNullptr) {
  DerivedClassRef ref1(tsl::MakeRef<DerivedClass>("test"));
  DerivedClassRef ref2;
  DerivedClassRef ref3;
  EXPECT_NE(ref1, ref2);
  EXPECT_NE(ref2, ref1);
  EXPECT_EQ(ref2, ref3);
}

TEST(RCReferenceWrapperTest, HashAndComparisonWithNonnull) {
  DerivedClassRef ref1(tsl::MakeRef<DerivedClass>("test"));
  DerivedClassRef ref2(tsl::MakeRef<DerivedClass>("test"));
  DerivedClassRef ref3(tsl::MakeRef<DerivedClass>("test3"));
  absl::flat_hash_set<BaseClassRef> set;
  set.insert(std::move(ref1));
  EXPECT_TRUE(set.contains(ref2));
  EXPECT_FALSE(set.contains(ref3));
}

TEST(RCReferenceWrapperTest, HashAndComparisonWithWrappedNullptr) {
  DerivedClassRef ref1;
  DerivedClassRef ref2;
  absl::flat_hash_set<BaseClassRef> set;
  set.insert(std::move(ref1));
  EXPECT_TRUE(set.contains(ref2));
}

TEST(RCReferenceWrapperTest, StringifyWithNonnull) {
  DerivedClassRef ref(tsl::MakeRef<DerivedClass>("test"));
  EXPECT_EQ(absl::StrCat(ref), "test");
}

TEST(RCReferenceWrapperTest, StringifyWithWrappedNullptr) {
  DerivedClassRef ref;
  EXPECT_EQ(absl::StrCat(ref), "<nullptr>");
}

TEST(RCReferenceWrapperTest, Reset) {
  DerivedClassRef ref(tsl::MakeRef<DerivedClass>("test"));
  ref.reset();
  EXPECT_EQ(ref, nullptr);
  EXPECT_EQ(nullptr, ref);
}

TEST(RCReferenceWrapperTest, Release) {
  DerivedClassRef ref1(tsl::MakeRef<DerivedClass>("test"));
  DerivedClassRef ref2(tsl::TakeRef(ref1.release()));
  EXPECT_EQ(ref1, nullptr);
  EXPECT_EQ(ref2->name(), "test");
}

TEST(RCReferenceWrapperTest, Swap) {
  DerivedClassRef ref1(tsl::MakeRef<DerivedClass>("test1"));
  DerivedClassRef ref2(tsl::MakeRef<DerivedClass>("test2"));
  ref1.swap(ref2);
  EXPECT_EQ(ref1->name(), "test2");
  EXPECT_EQ(ref2->name(), "test1");
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
