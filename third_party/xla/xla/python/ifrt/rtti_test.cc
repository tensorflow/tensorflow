/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/rtti.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/python/ifrt/ref_wrapper.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::IsNull;

// Regular classes that do not use `tsl::RCReference`.

class Base : public RTTIExtends<Base, RTTIRoot> {
 public:
  static char ID;  // NOLINT
};

class DerivedA : public RTTIExtends<DerivedA, Base> {
 public:
  static char ID;  // NOLINT
};

class DerivedB : public RTTIExtends<DerivedB, Base> {
 public:
  static char ID;  // NOLINT
};

class Leaf : public RTTIExtends<Leaf, DerivedA> {
 public:
  static char ID;  // NOLINT
};

[[maybe_unused]] char Base::ID = 0;
[[maybe_unused]] char DerivedA::ID = 0;
[[maybe_unused]] char DerivedB::ID = 0;
[[maybe_unused]] char Leaf::ID = 0;

// Classes that use `tsl::RCReference`.

class RefBase : public tsl::ReferenceCounted<RefBase>,
                public RTTIExtends<RefBase, RTTIRoot> {
 public:
  static char ID;  // NOLINT
};

class RefDerivedA : public RTTIExtends<RefDerivedA, RefBase> {
 public:
  static char ID;  // NOLINT
};

class RefDerivedB : public RTTIExtends<RefDerivedB, RefBase> {
 public:
  static char ID;  // NOLINT
};

class RefLeaf : public RTTIExtends<RefLeaf, RefDerivedA> {
 public:
  static char ID;  // NOLINT
};

[[maybe_unused]] char RefBase::ID = 0;
[[maybe_unused]] char RefDerivedA::ID = 0;
[[maybe_unused]] char RefDerivedB::ID = 0;
[[maybe_unused]] char RefLeaf::ID = 0;

TEST(RttiTest, ClassIdAndDynamicClassIdRegular) {
  Base base;
  DerivedA derived_a;
  DerivedB derived_b;
  Leaf leaf;
  Base* derived_a_base = &derived_a;
  Base* leaf_base = &leaf;
  EXPECT_NE(Base::classID(), DerivedA::classID());
  EXPECT_NE(DerivedA::classID(), DerivedB::classID());
  EXPECT_EQ(base.dynamicClassID(), Base::classID());
  EXPECT_EQ(derived_a.dynamicClassID(), DerivedA::classID());
  EXPECT_EQ(derived_b.dynamicClassID(), DerivedB::classID());
  EXPECT_EQ(leaf.dynamicClassID(), Leaf::classID());
  EXPECT_EQ(derived_a_base->dynamicClassID(), DerivedA::classID());
  EXPECT_EQ(leaf_base->dynamicClassID(), Leaf::classID());
}

TEST(RttiTest, ClassIdAndDynamicClassIdRCReference) {
  tsl::RCReference<RefBase> rcref_base = tsl::MakeRef<RefBase>();
  tsl::RCReference<RefDerivedA> rcref_derived_a = tsl::MakeRef<RefDerivedA>();
  tsl::RCReference<RefDerivedB> rcref_derived_b = tsl::MakeRef<RefDerivedB>();
  tsl::RCReference<RefLeaf> rcref_leaf = tsl::MakeRef<RefLeaf>();
  tsl::RCReference<RefBase> rcref_derived_a_base(rcref_derived_a);
  tsl::RCReference<RefBase> rcref_leaf_base(rcref_leaf);
  EXPECT_NE(RefBase::classID(), RefDerivedA::classID());
  EXPECT_NE(RefDerivedA::classID(), RefDerivedB::classID());
  EXPECT_EQ(rcref_base->dynamicClassID(), RefBase::classID());
  EXPECT_EQ(rcref_derived_a->dynamicClassID(), RefDerivedA::classID());
  EXPECT_EQ(rcref_derived_b->dynamicClassID(), RefDerivedB::classID());
  EXPECT_EQ(rcref_leaf->dynamicClassID(), RefLeaf::classID());
  EXPECT_EQ(rcref_derived_a_base->dynamicClassID(), RefDerivedA::classID());
  EXPECT_EQ(rcref_leaf_base->dynamicClassID(), RefLeaf::classID());
}

TEST(RttiTest, DynamicClassIDUponConstructionOrAssignment) {
  DerivedA original;
  EXPECT_EQ(original.dynamicClassID(), DerivedA::classID());

  DerivedA copy_constructed(original);
  EXPECT_EQ(copy_constructed.dynamicClassID(), DerivedA::classID());
  EXPECT_TRUE(isa<DerivedA>(copy_constructed));

  DerivedA move_constructed(std::move(copy_constructed));
  EXPECT_EQ(move_constructed.dynamicClassID(), DerivedA::classID());
  EXPECT_TRUE(isa<DerivedA>(move_constructed));

  DerivedA copy_assigned;
  copy_assigned = original;
  EXPECT_EQ(copy_assigned.dynamicClassID(), DerivedA::classID());
  EXPECT_TRUE(isa<DerivedA>(copy_assigned));

  DerivedA move_assigned;
  move_assigned = std::move(copy_assigned);
  EXPECT_EQ(move_assigned.dynamicClassID(), DerivedA::classID());
  EXPECT_TRUE(isa<DerivedA>(move_assigned));
}

TEST(RttiTest, IsARegular) {
  auto test = [](const auto& derived_a, const auto& leaf) {
    EXPECT_TRUE(isa<RTTIRoot>(derived_a));
    EXPECT_TRUE(isa<Base>(derived_a));
    EXPECT_TRUE(isa<DerivedA>(derived_a));
    EXPECT_FALSE(isa<DerivedB>(derived_a));
    EXPECT_FALSE(isa<Leaf>(derived_a));

    EXPECT_TRUE(isa<RTTIRoot>(leaf));
    EXPECT_TRUE(isa<Base>(leaf));
    EXPECT_TRUE(isa<DerivedA>(leaf));
    EXPECT_FALSE(isa<DerivedB>(leaf));
    EXPECT_TRUE(isa<Leaf>(leaf));

    EXPECT_TRUE((isa<DerivedA, DerivedB>(derived_a)));
    EXPECT_TRUE((isa<DerivedB, DerivedA>(derived_a)));
    EXPECT_FALSE((isa<DerivedB, Leaf>(derived_a)));

    EXPECT_TRUE((isa<DerivedA, DerivedB>(leaf)));
    EXPECT_TRUE((isa<DerivedB, Leaf>(leaf)));
    EXPECT_FALSE((isa<DerivedB, DerivedB>(leaf)));
  };

  DerivedA derived_a;
  Leaf leaf;
  Base* ptr_derived_a = &derived_a;
  Base* ptr_leaf = &leaf;
  test(ptr_derived_a, ptr_leaf);

  Base& ref_derived_a = derived_a;
  Base& ref_leaf = leaf;
  test(ref_derived_a, ref_leaf);

  std::unique_ptr<Base> unique_derived_a = std::make_unique<DerivedA>();
  std::unique_ptr<Base> unique_leaf = std::make_unique<Leaf>();
  test(unique_derived_a, unique_leaf);

  std::shared_ptr<Base> shared_derived_a = std::make_shared<DerivedA>();
  std::shared_ptr<Base> shared_leaf = std::make_shared<Leaf>();
  test(shared_derived_a, shared_leaf);
}

TEST(RttiTest, IsARCReference) {
  auto test = [](const auto& derived_a, const auto& leaf) {
    EXPECT_TRUE(isa<RTTIRoot>(derived_a));
    EXPECT_TRUE(isa<RefBase>(derived_a));
    EXPECT_TRUE(isa<RefDerivedA>(derived_a));
    EXPECT_FALSE(isa<RefDerivedB>(derived_a));
    EXPECT_FALSE(isa<RefLeaf>(derived_a));

    EXPECT_TRUE(isa<RTTIRoot>(leaf));
    EXPECT_TRUE(isa<RefBase>(leaf));
    EXPECT_TRUE(isa<RefDerivedA>(leaf));
    EXPECT_FALSE(isa<RefDerivedB>(leaf));
    EXPECT_TRUE(isa<RefLeaf>(leaf));

    EXPECT_TRUE((isa<RefDerivedA, RefDerivedB>(derived_a)));
    EXPECT_TRUE((isa<RefDerivedB, RefDerivedA>(derived_a)));
    EXPECT_FALSE((isa<RefDerivedB, RefLeaf>(derived_a)));

    EXPECT_TRUE((isa<RefDerivedA, RefDerivedB>(leaf)));
    EXPECT_TRUE((isa<RefDerivedB, RefLeaf>(leaf)));
    EXPECT_FALSE((isa<RefDerivedB, RefDerivedB>(leaf)));
  };

  tsl::RCReference<RefBase> rcref_derived_a = tsl::MakeRef<RefDerivedA>();
  tsl::RCReference<RefBase> rcref_leaf = tsl::MakeRef<RefLeaf>();
  test(rcref_derived_a, rcref_leaf);

  RCReferenceWrapper<RefBase> rcrefwrapper_derived_a(
      tsl::MakeRef<RefDerivedA>());
  RCReferenceWrapper<RefBase> rcrefwrapper_leaf(tsl::MakeRef<RefLeaf>());
  test(rcrefwrapper_derived_a, rcrefwrapper_leaf);
}

TEST(RttiTest, IsAPresentAndNonNullRegular) {
  auto test = [](const auto& derived_a, const auto& leaf, const auto& null) {
    EXPECT_TRUE(isa_and_present<RTTIRoot>(derived_a));
    EXPECT_TRUE(isa_and_present<Base>(derived_a));
    EXPECT_TRUE(isa_and_present<DerivedA>(derived_a));
    EXPECT_FALSE(isa_and_present<DerivedB>(derived_a));
    EXPECT_FALSE(isa_and_present<Leaf>(derived_a));

    EXPECT_TRUE(isa_and_present<RTTIRoot>(leaf));
    EXPECT_TRUE(isa_and_present<Base>(leaf));
    EXPECT_TRUE(isa_and_present<DerivedA>(leaf));
    EXPECT_FALSE(isa_and_present<DerivedB>(leaf));
    EXPECT_TRUE(isa_and_present<Leaf>(leaf));

    EXPECT_FALSE(isa_and_present<RTTIRoot>(null));
    EXPECT_FALSE(isa_and_present<Base>(null));
    EXPECT_FALSE(isa_and_present<DerivedA>(null));
    EXPECT_FALSE(isa_and_present<DerivedB>(null));
    EXPECT_FALSE(isa_and_present<Leaf>(null));

    EXPECT_TRUE(isa_and_nonnull<RTTIRoot>(derived_a));
    EXPECT_TRUE(isa_and_nonnull<Base>(derived_a));
    EXPECT_TRUE(isa_and_nonnull<DerivedA>(derived_a));
    EXPECT_FALSE(isa_and_nonnull<DerivedB>(derived_a));
    EXPECT_FALSE(isa_and_nonnull<Leaf>(derived_a));

    EXPECT_TRUE(isa_and_nonnull<RTTIRoot>(leaf));
    EXPECT_TRUE(isa_and_nonnull<Base>(leaf));
    EXPECT_TRUE(isa_and_nonnull<DerivedA>(leaf));
    EXPECT_FALSE(isa_and_nonnull<DerivedB>(leaf));
    EXPECT_TRUE(isa_and_nonnull<Leaf>(leaf));

    EXPECT_FALSE(isa_and_nonnull<RTTIRoot>(null));
    EXPECT_FALSE(isa_and_nonnull<Base>(null));
    EXPECT_FALSE(isa_and_nonnull<DerivedA>(null));
    EXPECT_FALSE(isa_and_nonnull<DerivedB>(null));
    EXPECT_FALSE(isa_and_nonnull<Leaf>(null));
  };

  DerivedA derived_a;
  Leaf leaf;
  Base* ptr_derived_a = &derived_a;
  Base* ptr_leaf = &leaf;
  Base* ptr_null = nullptr;
  test(ptr_derived_a, ptr_leaf, ptr_null);

  std::unique_ptr<Base> unique_derived_a = std::make_unique<DerivedA>();
  std::unique_ptr<Base> unique_leaf = std::make_unique<Leaf>();
  std::unique_ptr<Base> unique_null;
  test(unique_derived_a, unique_leaf, unique_null);

  std::shared_ptr<Base> shared_derived_a = std::make_shared<DerivedA>();
  std::shared_ptr<Base> shared_leaf = std::make_shared<Leaf>();
  std::shared_ptr<Base> shared_null;
  test(shared_derived_a, shared_leaf, shared_null);
}

TEST(RttiTest, IsAPresentAndNonNullRCReference) {
  auto test = [](const auto& derived_a, const auto& leaf, const auto& null) {
    EXPECT_TRUE(isa_and_present<RTTIRoot>(derived_a));
    EXPECT_TRUE(isa_and_present<RefBase>(derived_a));
    EXPECT_TRUE(isa_and_present<RefDerivedA>(derived_a));
    EXPECT_FALSE(isa_and_present<RefDerivedB>(derived_a));
    EXPECT_FALSE(isa_and_present<RefLeaf>(derived_a));

    EXPECT_TRUE(isa_and_present<RTTIRoot>(leaf));
    EXPECT_TRUE(isa_and_present<RefBase>(leaf));
    EXPECT_TRUE(isa_and_present<RefDerivedA>(leaf));
    EXPECT_FALSE(isa_and_present<RefDerivedB>(leaf));
    EXPECT_TRUE(isa_and_present<RefLeaf>(leaf));

    EXPECT_FALSE(isa_and_present<RTTIRoot>(null));
    EXPECT_FALSE(isa_and_present<RefBase>(null));
    EXPECT_FALSE(isa_and_present<RefDerivedA>(null));
    EXPECT_FALSE(isa_and_present<RefDerivedB>(null));
    EXPECT_FALSE(isa_and_present<RefLeaf>(null));

    EXPECT_TRUE(isa_and_nonnull<RTTIRoot>(derived_a));
    EXPECT_TRUE(isa_and_nonnull<RefBase>(derived_a));
    EXPECT_TRUE(isa_and_nonnull<RefDerivedA>(derived_a));
    EXPECT_FALSE(isa_and_nonnull<RefDerivedB>(derived_a));
    EXPECT_FALSE(isa_and_nonnull<RefLeaf>(derived_a));

    EXPECT_TRUE(isa_and_nonnull<RTTIRoot>(leaf));
    EXPECT_TRUE(isa_and_nonnull<RefBase>(leaf));
    EXPECT_TRUE(isa_and_nonnull<RefDerivedA>(leaf));
    EXPECT_FALSE(isa_and_nonnull<RefDerivedB>(leaf));
    EXPECT_TRUE(isa_and_nonnull<RefLeaf>(leaf));

    EXPECT_FALSE(isa_and_nonnull<RTTIRoot>(null));
    EXPECT_FALSE(isa_and_nonnull<RefBase>(null));
    EXPECT_FALSE(isa_and_nonnull<RefDerivedA>(null));
    EXPECT_FALSE(isa_and_nonnull<RefDerivedB>(null));
    EXPECT_FALSE(isa_and_nonnull<RefLeaf>(null));
  };

  tsl::RCReference<RefBase> rcref_derived_a = tsl::MakeRef<RefDerivedA>();
  tsl::RCReference<RefBase> rcref_leaf = tsl::MakeRef<RefLeaf>();
  tsl::RCReference<RefBase> rcref_null;
  test(rcref_derived_a, rcref_leaf, rcref_null);

  RCReferenceWrapper<RefBase> rcrefwrapper_derived_a(
      tsl::MakeRef<RefDerivedA>());
  RCReferenceWrapper<RefBase> rcrefwrapper_leaf(tsl::MakeRef<RefLeaf>());
  RCReferenceWrapper<RefBase> rcrefwrapper_null;
  test(rcrefwrapper_derived_a, rcrefwrapper_leaf, rcrefwrapper_null);
}

TEST(RttiTest, CastRegular) {
  auto test_pass_by_ptr = [](const auto* derived_a, const auto* leaf) {
    EXPECT_EQ(cast<DerivedA>(derived_a), derived_a);
    EXPECT_EQ(cast<DerivedA>(leaf), leaf);
    EXPECT_EQ(cast<Leaf>(leaf), leaf);
  };
  auto test_pass_by_ref = [](const auto& derived_a, const auto& leaf) {
    EXPECT_EQ(&cast<DerivedA>(derived_a), &derived_a);
    EXPECT_EQ(&cast<DerivedA>(leaf), &leaf);
    EXPECT_EQ(&cast<Leaf>(leaf), &leaf);
  };
  auto test_pass_by_smart_ptr_ref = [](const auto& derived_a,
                                       const auto& leaf) {
    EXPECT_EQ(cast<DerivedA>(derived_a).get(), derived_a.get());
    EXPECT_EQ(cast<DerivedA>(leaf).get(), leaf.get());
    EXPECT_EQ(cast<Leaf>(leaf).get(), leaf.get());
  };
  auto test_pass_by_smart_ptr_value = [](auto derived_a, auto leaf) {
    const auto* derived_a_ptr = derived_a.get();
    const auto* leaf_ptr = leaf.get();
    EXPECT_EQ(cast<DerivedA>(std::move(derived_a)).get(), derived_a_ptr);
    EXPECT_EQ(cast<DerivedA>(std::move(leaf)).get(), leaf_ptr);
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(derived_a.get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(leaf.get(), IsNull());
  };

  DerivedA derived_a;
  Leaf leaf;
  Base* ptr_derived_a = &derived_a;
  Base* ptr_leaf = &leaf;
  test_pass_by_ptr(ptr_derived_a, ptr_leaf);

  Base& ref_derived_a = derived_a;
  Base& ref_leaf = leaf;
  test_pass_by_ref(ref_derived_a, ref_leaf);

  std::unique_ptr<Base> unique_derived_a = std::make_unique<DerivedA>();
  std::unique_ptr<Base> unique_leaf = std::make_unique<Leaf>();
  test_pass_by_smart_ptr_value(std::move(unique_derived_a),
                               std::move(unique_leaf));

  std::shared_ptr<Base> shared_derived_a = std::make_shared<DerivedA>();
  std::shared_ptr<Base> shared_leaf = std::make_shared<Leaf>();
  test_pass_by_smart_ptr_ref(shared_derived_a, shared_leaf);
  test_pass_by_smart_ptr_value(std::move(shared_derived_a),
                               std::move(shared_leaf));
}

TEST(RttiTest, CastRCReference) {
  auto test_pass_by_smart_ptr_ref = [](const auto& derived_a,
                                       const auto& leaf) {
    EXPECT_EQ(cast<RefDerivedA>(derived_a).get(), derived_a.get());
    EXPECT_EQ(cast<RefDerivedA>(leaf).get(), leaf.get());
    EXPECT_EQ(cast<RefLeaf>(leaf).get(), leaf.get());
  };
  auto test_pass_by_smart_ptr_value = [](auto derived_a, auto leaf) {
    const auto* derived_a_ptr = derived_a.get();
    const auto* leaf_ptr = leaf.get();
    EXPECT_EQ(cast<RefDerivedA>(std::move(derived_a)).get(), derived_a_ptr);
    EXPECT_EQ(cast<RefDerivedA>(std::move(leaf)).get(), leaf_ptr);
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(derived_a.get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(leaf.get(), IsNull());
  };

  tsl::RCReference<RefBase> rcref_derived_a = tsl::MakeRef<RefDerivedA>();
  tsl::RCReference<RefBase> rcref_leaf = tsl::MakeRef<RefLeaf>();
  test_pass_by_smart_ptr_ref(rcref_derived_a, rcref_leaf);
  test_pass_by_smart_ptr_value(std::move(rcref_derived_a),
                               std::move(rcref_leaf));

  RCReferenceWrapper<RefBase> rcrefwrapper_derived_a(
      tsl::MakeRef<RefDerivedA>());
  RCReferenceWrapper<RefBase> rcrefwrapper_leaf(tsl::MakeRef<RefLeaf>());
  test_pass_by_smart_ptr_ref(rcrefwrapper_derived_a, rcrefwrapper_leaf);
  test_pass_by_smart_ptr_value(std::move(rcrefwrapper_derived_a),
                               std::move(rcrefwrapper_leaf));
}

TEST(RttiTest, CastIfPresentRegular) {
  auto test_pass_by_ptr = [](const auto* derived_a, const auto* leaf,
                             const auto* null) {
    EXPECT_EQ(cast_if_present<DerivedA>(derived_a), derived_a);
    EXPECT_EQ(cast_if_present<DerivedA>(leaf), leaf);
    EXPECT_EQ(cast_if_present<Leaf>(leaf), leaf);
    EXPECT_THAT(cast_if_present<DerivedA>(null), IsNull());

    EXPECT_EQ(cast_or_null<DerivedA>(derived_a), derived_a);
    EXPECT_EQ(cast_or_null<DerivedA>(leaf), leaf);
    EXPECT_EQ(cast_or_null<Leaf>(leaf), leaf);
    EXPECT_THAT(cast_or_null<DerivedA>(null), IsNull());
  };
  auto test_pass_by_smart_ptr_ref = [](const auto& derived_a, const auto& leaf,
                                       const auto& null) {
    EXPECT_EQ(cast_if_present<DerivedA>(derived_a).get(), derived_a.get());
    EXPECT_EQ(cast_if_present<DerivedA>(leaf).get(), leaf.get());
    EXPECT_EQ(cast_if_present<Leaf>(leaf).get(), leaf.get());
    EXPECT_THAT(cast_if_present<DerivedA>(null).get(), IsNull());

    EXPECT_EQ(cast_or_null<DerivedA>(derived_a).get(), derived_a.get());
    EXPECT_EQ(cast_or_null<DerivedA>(leaf).get(), leaf.get());
    EXPECT_EQ(cast_or_null<Leaf>(leaf).get(), leaf.get());
    EXPECT_THAT(cast_or_null<DerivedA>(null).get(), IsNull());
  };
  auto test_pass_by_smart_ptr_value = [](auto derived_a, auto leaf, auto null) {
    const auto* derived_a_ptr = derived_a.get();
    const auto* leaf_ptr = leaf.get();
    EXPECT_EQ(cast_if_present<DerivedA>(std::move(derived_a)).get(),
              derived_a_ptr);
    EXPECT_EQ(cast_if_present<DerivedA>(std::move(leaf)).get(), leaf_ptr);
    EXPECT_THAT(cast_if_present<DerivedA>(std::move(null)).get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(derived_a.get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(leaf.get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(null.get(), IsNull());
  };

  DerivedA derived_a;
  Leaf leaf;
  Base* ptr_derived_a = &derived_a;
  Base* ptr_leaf = &leaf;
  Base* ptr_null = nullptr;
  test_pass_by_ptr(ptr_derived_a, ptr_leaf, ptr_null);

  std::unique_ptr<Base> unique_derived_a = std::make_unique<DerivedA>();
  std::unique_ptr<Base> unique_leaf = std::make_unique<Leaf>();
  std::unique_ptr<Base> unique_null;
  test_pass_by_smart_ptr_value(std::move(unique_derived_a),
                               std::move(unique_leaf), std::move(unique_null));

  std::shared_ptr<Base> shared_derived_a = std::make_shared<DerivedA>();
  std::shared_ptr<Base> shared_leaf = std::make_shared<Leaf>();
  std::shared_ptr<Base> shared_null;
  test_pass_by_smart_ptr_ref(shared_derived_a, shared_leaf, shared_null);
  test_pass_by_smart_ptr_value(std::move(shared_derived_a),
                               std::move(shared_leaf), std::move(shared_null));
}

TEST(RttiTest, CastIfPresentRCReference) {
  auto test_pass_by_smart_ptr_ref = [](const auto& derived_a, const auto& leaf,
                                       const auto& null) {
    EXPECT_EQ(cast_if_present<RefDerivedA>(derived_a).get(), derived_a.get());
    EXPECT_EQ(cast_if_present<RefDerivedA>(leaf).get(), leaf.get());
    EXPECT_EQ(cast_if_present<RefLeaf>(leaf).get(), leaf.get());
    EXPECT_THAT(cast_if_present<RefDerivedA>(null).get(), IsNull());

    EXPECT_EQ(cast_or_null<RefDerivedA>(derived_a).get(), derived_a.get());
    EXPECT_EQ(cast_or_null<RefDerivedA>(leaf).get(), leaf.get());
    EXPECT_EQ(cast_or_null<RefLeaf>(leaf).get(), leaf.get());
    EXPECT_THAT(cast_or_null<RefDerivedA>(null).get(), IsNull());
  };
  auto test_pass_by_smart_ptr_value = [](auto derived_a, auto leaf, auto null) {
    const auto* derived_a_ptr = derived_a.get();
    const auto* leaf_ptr = leaf.get();
    EXPECT_EQ(cast_if_present<RefDerivedA>(std::move(derived_a)).get(),
              derived_a_ptr);
    EXPECT_EQ(cast_if_present<RefDerivedA>(std::move(leaf)).get(), leaf_ptr);
    EXPECT_THAT(cast_if_present<RefDerivedA>(std::move(null)).get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(derived_a.get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(leaf.get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(null.get(), IsNull());
  };

  tsl::RCReference<RefBase> rcref_derived_a = tsl::MakeRef<RefDerivedA>();
  tsl::RCReference<RefBase> rcref_leaf = tsl::MakeRef<RefLeaf>();
  tsl::RCReference<RefBase> rcref_null;
  test_pass_by_smart_ptr_ref(rcref_derived_a, rcref_leaf, rcref_null);
  test_pass_by_smart_ptr_value(std::move(rcref_derived_a),
                               std::move(rcref_leaf), std::move(rcref_null));

  RCReferenceWrapper<RefBase> rcrefwrapper_derived_a(
      tsl::MakeRef<RefDerivedA>());
  RCReferenceWrapper<RefBase> rcrefwrapper_leaf(tsl::MakeRef<RefLeaf>());
  RCReferenceWrapper<RefBase> rcrefwrapper_null;
  test_pass_by_smart_ptr_ref(rcrefwrapper_derived_a, rcrefwrapper_leaf,
                             rcrefwrapper_null);
  test_pass_by_smart_ptr_value(std::move(rcrefwrapper_derived_a),
                               std::move(rcrefwrapper_leaf),
                               std::move(rcrefwrapper_null));
}

TEST(RttiTest, DynCastRegular) {
  auto test_pass_by_ptr = [](const auto* derived_a, const auto* leaf) {
    EXPECT_EQ(dyn_cast<DerivedA>(derived_a), derived_a);
    EXPECT_THAT(dyn_cast<DerivedB>(derived_a), IsNull());
    EXPECT_EQ(dyn_cast<DerivedA>(leaf), leaf);
    EXPECT_THAT(dyn_cast<DerivedB>(leaf), IsNull());
    EXPECT_EQ(dyn_cast<Leaf>(leaf), leaf);
    EXPECT_THAT(dyn_cast<Leaf>(derived_a), IsNull());
  };
  auto test_pass_by_smart_ptr_ref = [](const auto& derived_a,
                                       const auto& leaf) {
    EXPECT_EQ(dyn_cast<DerivedA>(derived_a).get(), derived_a.get());
    EXPECT_THAT(dyn_cast<DerivedB>(derived_a).get(), IsNull());
    EXPECT_EQ(dyn_cast<DerivedA>(leaf).get(), leaf.get());
    EXPECT_THAT(dyn_cast<DerivedB>(leaf).get(), IsNull());
    EXPECT_EQ(dyn_cast<Leaf>(leaf).get(), leaf.get());
    EXPECT_THAT(dyn_cast<Leaf>(derived_a).get(), IsNull());
  };
  auto test_pass_by_smart_ptr_value = [](auto derived_a, auto leaf) {
    const auto* derived_a_ptr = derived_a.get();
    const auto* leaf_ptr = leaf.get();
    EXPECT_EQ(dyn_cast<DerivedA>(std::move(derived_a)).get(), derived_a_ptr);
    EXPECT_EQ(dyn_cast<DerivedA>(std::move(leaf)).get(), leaf_ptr);
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(derived_a.get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(leaf.get(), IsNull());
  };

  DerivedA derived_a;
  Leaf leaf;
  Base* ptr_derived_a = &derived_a;
  Base* ptr_leaf = &leaf;
  test_pass_by_ptr(ptr_derived_a, ptr_leaf);

  std::unique_ptr<Base> unique_derived_a = std::make_unique<DerivedA>();
  std::unique_ptr<Base> unique_leaf = std::make_unique<Leaf>();
  test_pass_by_smart_ptr_value(std::move(unique_derived_a),
                               std::move(unique_leaf));

  std::shared_ptr<Base> shared_derived_a = std::make_shared<DerivedA>();
  std::shared_ptr<Base> shared_leaf = std::make_shared<Leaf>();
  test_pass_by_smart_ptr_ref(shared_derived_a, shared_leaf);
  test_pass_by_smart_ptr_value(std::move(shared_derived_a),
                               std::move(shared_leaf));
}

TEST(RttiTest, DynCastRCReference) {
  auto test_pass_by_smart_ptr_ref = [](const auto& derived_a,
                                       const auto& leaf) {
    EXPECT_EQ(dyn_cast<RefDerivedA>(derived_a).get(), derived_a.get());
    EXPECT_THAT(dyn_cast<RefDerivedB>(derived_a).get(), IsNull());
    EXPECT_EQ(dyn_cast<RefDerivedA>(leaf).get(), leaf.get());
    EXPECT_THAT(dyn_cast<RefDerivedB>(leaf).get(), IsNull());
    EXPECT_EQ(dyn_cast<RefLeaf>(leaf).get(), leaf.get());
    EXPECT_THAT(dyn_cast<RefLeaf>(derived_a).get(), IsNull());
  };
  auto test_pass_by_smart_ptr_value = [](auto derived_a, auto leaf) {
    const auto* derived_a_ptr = derived_a.get();
    const auto* leaf_ptr = leaf.get();
    EXPECT_EQ(dyn_cast<RefDerivedA>(std::move(derived_a)).get(), derived_a_ptr);
    EXPECT_EQ(dyn_cast<RefDerivedA>(std::move(leaf)).get(), leaf_ptr);
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(derived_a.get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(leaf.get(), IsNull());
  };

  tsl::RCReference<RefBase> rcref_derived_a = tsl::MakeRef<RefDerivedA>();
  tsl::RCReference<RefBase> rcref_leaf = tsl::MakeRef<RefLeaf>();
  test_pass_by_smart_ptr_ref(rcref_derived_a, rcref_leaf);
  test_pass_by_smart_ptr_value(std::move(rcref_derived_a),
                               std::move(rcref_leaf));

  RCReferenceWrapper<RefBase> rcrefwrapper_derived_a(
      tsl::MakeRef<RefDerivedA>());
  RCReferenceWrapper<RefBase> rcrefwrapper_leaf(tsl::MakeRef<RefLeaf>());
  test_pass_by_smart_ptr_ref(rcrefwrapper_derived_a, rcrefwrapper_leaf);
  test_pass_by_smart_ptr_value(std::move(rcrefwrapper_derived_a),
                               std::move(rcrefwrapper_leaf));
}

TEST(RttiTest, DynCastIfPresentRegular) {
  auto test_pass_by_ptr = [](const auto* derived_a, const auto* leaf,
                             const auto* null) {
    EXPECT_EQ(dyn_cast_if_present<DerivedA>(derived_a), derived_a);
    EXPECT_THAT(dyn_cast_if_present<DerivedB>(derived_a), IsNull());
    EXPECT_EQ(dyn_cast_if_present<DerivedA>(leaf), leaf);
    EXPECT_THAT(dyn_cast_if_present<DerivedB>(leaf), IsNull());
    EXPECT_EQ(dyn_cast_if_present<Leaf>(leaf), leaf);
    EXPECT_THAT(dyn_cast_if_present<Leaf>(derived_a), IsNull());
    EXPECT_THAT(dyn_cast_if_present<DerivedA>(null), IsNull());

    EXPECT_EQ(dyn_cast_or_null<DerivedA>(derived_a), derived_a);
    EXPECT_THAT(dyn_cast_or_null<DerivedB>(derived_a), IsNull());
    EXPECT_EQ(dyn_cast_or_null<DerivedA>(leaf), leaf);
    EXPECT_THAT(dyn_cast_or_null<DerivedB>(leaf), IsNull());
    EXPECT_EQ(dyn_cast_or_null<Leaf>(leaf), leaf);
    EXPECT_THAT(dyn_cast_or_null<Leaf>(derived_a), IsNull());
    EXPECT_THAT(dyn_cast_or_null<DerivedA>(null), IsNull());
  };
  auto test_pass_by_smart_ptr_ref = [](const auto& derived_a, const auto& leaf,
                                       const auto& null) {
    EXPECT_EQ(dyn_cast_if_present<DerivedA>(derived_a).get(), derived_a.get());
    EXPECT_THAT(dyn_cast_if_present<DerivedB>(derived_a).get(), IsNull());
    EXPECT_EQ(dyn_cast_if_present<DerivedA>(leaf).get(), leaf.get());
    EXPECT_THAT(dyn_cast_if_present<DerivedB>(leaf).get(), IsNull());
    EXPECT_EQ(dyn_cast_if_present<Leaf>(leaf).get(), leaf.get());
    EXPECT_THAT(dyn_cast_if_present<Leaf>(derived_a).get(), IsNull());
    EXPECT_THAT(dyn_cast_if_present<DerivedA>(null).get(), IsNull());

    EXPECT_EQ(dyn_cast_or_null<DerivedA>(derived_a).get(), derived_a.get());
    EXPECT_THAT(dyn_cast_or_null<DerivedB>(derived_a).get(), IsNull());
    EXPECT_EQ(dyn_cast_or_null<DerivedA>(leaf).get(), leaf.get());
    EXPECT_THAT(dyn_cast_or_null<DerivedB>(leaf).get(), IsNull());
    EXPECT_EQ(dyn_cast_or_null<Leaf>(leaf).get(), leaf.get());
    EXPECT_THAT(dyn_cast_or_null<Leaf>(derived_a).get(), IsNull());
    EXPECT_THAT(dyn_cast_or_null<DerivedA>(null).get(), IsNull());
  };
  auto test_pass_by_smart_ptr_value = [](auto derived_a, auto leaf, auto null) {
    const auto* derived_a_ptr = derived_a.get();
    const auto* leaf_ptr = leaf.get();
    EXPECT_EQ(dyn_cast_if_present<DerivedA>(std::move(derived_a)).get(),
              derived_a_ptr);
    EXPECT_EQ(dyn_cast_if_present<DerivedA>(std::move(leaf)).get(), leaf_ptr);
    EXPECT_THAT(dyn_cast_if_present<DerivedA>(std::move(null)).get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(derived_a.get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(leaf.get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(null.get(), IsNull());
  };

  DerivedA derived_a;
  Leaf leaf;
  Base* ptr_derived_a = &derived_a;
  Base* ptr_leaf = &leaf;
  Base* ptr_null = nullptr;
  test_pass_by_ptr(ptr_derived_a, ptr_leaf, ptr_null);

  std::unique_ptr<Base> unique_derived_a = std::make_unique<DerivedA>();
  std::unique_ptr<Base> unique_leaf = std::make_unique<Leaf>();
  std::unique_ptr<Base> unique_null;
  test_pass_by_smart_ptr_value(std::move(unique_derived_a),
                               std::move(unique_leaf), std::move(unique_null));

  std::shared_ptr<Base> shared_derived_a = std::make_shared<DerivedA>();
  std::shared_ptr<Base> shared_leaf = std::make_shared<Leaf>();
  std::shared_ptr<Base> shared_null;
  test_pass_by_smart_ptr_ref(shared_derived_a, shared_leaf, shared_null);
  test_pass_by_smart_ptr_value(std::move(shared_derived_a),
                               std::move(shared_leaf), std::move(shared_null));
}

TEST(RttiTest, DynCastIfPresentRCReference) {
  auto test_pass_by_smart_ptr_ref = [](const auto& derived_a, const auto& leaf,
                                       const auto& null) {
    EXPECT_EQ(dyn_cast_if_present<RefDerivedA>(derived_a).get(),
              derived_a.get());
    EXPECT_THAT(dyn_cast_if_present<RefDerivedB>(derived_a).get(), IsNull());
    EXPECT_EQ(dyn_cast_if_present<RefDerivedA>(leaf).get(), leaf.get());
    EXPECT_THAT(dyn_cast_if_present<RefDerivedB>(leaf).get(), IsNull());
    EXPECT_EQ(dyn_cast_if_present<RefLeaf>(leaf).get(), leaf.get());
    EXPECT_THAT(dyn_cast_if_present<RefLeaf>(derived_a).get(), IsNull());
    EXPECT_THAT(dyn_cast_if_present<RefDerivedA>(null).get(), IsNull());

    EXPECT_EQ(dyn_cast_or_null<RefDerivedA>(derived_a).get(), derived_a.get());
    EXPECT_THAT(dyn_cast_or_null<RefDerivedB>(derived_a).get(), IsNull());
    EXPECT_EQ(dyn_cast_or_null<RefDerivedA>(leaf).get(), leaf.get());
    EXPECT_THAT(dyn_cast_or_null<RefDerivedB>(leaf).get(), IsNull());
    EXPECT_EQ(dyn_cast_or_null<RefLeaf>(leaf).get(), leaf.get());
    EXPECT_THAT(dyn_cast_or_null<RefLeaf>(derived_a).get(), IsNull());
    EXPECT_THAT(dyn_cast_or_null<RefDerivedA>(null).get(), IsNull());
  };
  auto test_pass_by_smart_ptr_value = [](auto derived_a, auto leaf, auto null) {
    const auto* derived_a_ptr = derived_a.get();
    const auto* leaf_ptr = leaf.get();
    EXPECT_EQ(dyn_cast_if_present<RefDerivedA>(std::move(derived_a)).get(),
              derived_a_ptr);
    EXPECT_EQ(dyn_cast_if_present<RefDerivedA>(std::move(leaf)).get(),
              leaf_ptr);
    EXPECT_THAT(dyn_cast_if_present<RefDerivedA>(std::move(null)).get(),
                IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(derived_a.get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(leaf.get(), IsNull());
    // NOLINTNEXTLINE(bugprone-use-after-move)
    EXPECT_THAT(null.get(), IsNull());
  };

  tsl::RCReference<RefBase> rcref_derived_a = tsl::MakeRef<RefDerivedA>();
  tsl::RCReference<RefBase> rcref_leaf = tsl::MakeRef<RefLeaf>();
  tsl::RCReference<RefBase> rcref_null;
  test_pass_by_smart_ptr_ref(rcref_derived_a, rcref_leaf, rcref_null);
  test_pass_by_smart_ptr_value(std::move(rcref_derived_a),
                               std::move(rcref_leaf), std::move(rcref_null));

  RCReferenceWrapper<RefBase> rcrefwrapper_derived_a(
      tsl::MakeRef<RefDerivedA>());
  RCReferenceWrapper<RefBase> rcrefwrapper_leaf(tsl::MakeRef<RefLeaf>());
  RCReferenceWrapper<RefBase> rcrefwrapper_null;
  test_pass_by_smart_ptr_ref(rcrefwrapper_derived_a, rcrefwrapper_leaf,
                             rcrefwrapper_null);
  test_pass_by_smart_ptr_value(std::move(rcrefwrapper_derived_a),
                               std::move(rcrefwrapper_leaf),
                               std::move(rcrefwrapper_null));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
