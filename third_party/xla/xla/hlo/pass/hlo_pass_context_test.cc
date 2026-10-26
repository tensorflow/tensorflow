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

#include "xla/hlo/pass/hlo_pass_context.h"

#include <type_traits>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"

namespace xla {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::Ref;

struct Foo {
  explicit Foo(int value = 0) : value(value) {}
  Foo(const Foo&) = delete;
  Foo& operator=(const Foo&) = delete;
  int value;
};
struct Bar {};
struct Base {
  virtual ~Base() = default;
};
struct Derived : Base {};

template <typename T, typename Arg, typename = void>
constexpr bool kCanSet = false;
template <typename T, typename Arg>
constexpr bool
    kCanSet<T, Arg,
            std::void_t<decltype(std::declval<HloPassContext&>().Set<T>(
                std::declval<Arg>()))>> = true;

// Temporaries would dangle, so Set() must not accept them.
static_assert(kCanSet<Foo, Foo&>);
static_assert(kCanSet<const Foo, const Foo&>);
static_assert(!kCanSet<Foo, Foo&&>);
static_assert(!kCanSet<const Foo, Foo&&>);

TEST(HloPassContextTest, GetsRegisteredObject) {
  Foo foo;
  HloPassContext ctx;
  ASSERT_OK(ctx.Set(foo));

  EXPECT_EQ(ctx.GetOrNull<Foo>(), &foo);
  EXPECT_THAT(ctx.Get<Foo>(), IsOkAndHolds(Ref(foo)));
}

TEST(HloPassContextTest, ReturnedReferenceIsMutable) {
  Foo foo;
  HloPassContext ctx;
  ASSERT_OK(ctx.Set(foo));

  ctx.GetOrNull<Foo>()->value = 42;

  EXPECT_EQ(foo.value, 42);
}

TEST(HloPassContextTest, MissingObject) {
  HloPassContext ctx;

  EXPECT_EQ(ctx.GetOrNull<Foo>(), nullptr);
  EXPECT_THAT(ctx.Get<Foo>(),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("HloPassContext has no xla::(anonymous "
                                 "namespace)::Foo")));
}

TEST(HloPassContextTest, TypesAreIndependent) {
  Foo foo;
  Bar bar;
  HloPassContext ctx;
  ASSERT_OK(ctx.Set(foo));
  ASSERT_OK(ctx.Set(bar));

  EXPECT_EQ(ctx.GetOrNull<Foo>(), &foo);
  EXPECT_EQ(ctx.GetOrNull<Bar>(), &bar);
  EXPECT_EQ(ctx.GetOrNull<int>(), nullptr);
}

TEST(HloPassContextTest, NonConstObjectIsRetrievableAsConst) {
  Foo foo;
  HloPassContext ctx;
  ASSERT_OK(ctx.Set(foo));

  EXPECT_EQ(ctx.GetOrNull<const Foo>(), &foo);
  EXPECT_THAT(ctx.Get<const Foo>(), IsOkAndHolds(Ref(foo)));
}

TEST(HloPassContextTest, ConstObjectIsNotRetrievableAsNonConst) {
  const Foo foo;
  HloPassContext ctx;
  ASSERT_OK(ctx.Set(foo));

  EXPECT_EQ(ctx.GetOrNull<const Foo>(), &foo);
  EXPECT_EQ(ctx.GetOrNull<Foo>(), nullptr);
  EXPECT_THAT(ctx.Get<Foo>(),
              StatusIs(absl::StatusCode::kNotFound, HasSubstr("non-const")));
}

TEST(HloPassContextTest, LooksUpExactType) {
  Derived derived;
  HloPassContext ctx;
  ASSERT_OK(ctx.Set(derived));

  EXPECT_EQ(ctx.GetOrNull<Base>(), nullptr);
  EXPECT_EQ(ctx.GetOrNull<Derived>(), &derived);
}

TEST(HloPassContextTest, RegistersAsExplicitlySpecifiedType) {
  Derived derived;
  HloPassContext ctx;
  ASSERT_OK(ctx.Set<Base>(derived));

  EXPECT_EQ(ctx.GetOrNull<Base>(), &derived);
  EXPECT_THAT(ctx.Get<Base>(), IsOkAndHolds(Ref(derived)));
  EXPECT_EQ(ctx.GetOrNull<Derived>(), nullptr);
}

TEST(HloPassContextTest, DuplicateSetFailsAndKeepsOriginal) {
  Foo foo1;
  Foo foo2;
  HloPassContext ctx;
  ASSERT_OK(ctx.Set(foo1));

  EXPECT_THAT(ctx.Set(foo2), StatusIs(absl::StatusCode::kAlreadyExists,
                                      HasSubstr("HloPassContext already has")));
  EXPECT_THAT(ctx.Set<const Foo>(foo2),
              StatusIs(absl::StatusCode::kAlreadyExists));
  EXPECT_EQ(ctx.GetOrNull<Foo>(), &foo1);
  EXPECT_EQ(ctx.GetOrNull<const Foo>(), &foo1);
}

TEST(HloPassContextTest, DuplicateSetOfConstThenNonConstFails) {
  const Foo foo1;
  Foo foo2;
  HloPassContext ctx;
  ASSERT_OK(ctx.Set(foo1));

  EXPECT_THAT(ctx.Set(foo2), StatusIs(absl::StatusCode::kAlreadyExists));
  EXPECT_EQ(ctx.GetOrNull<Foo>(), nullptr);
  EXPECT_EQ(ctx.GetOrNull<const Foo>(), &foo1);
}

absl::StatusOr<int> ReadFooValue(const HloPassContext& ctx) {
  ABSL_ASSIGN_OR_RETURN(const Foo& foo, ctx.Get<const Foo>());
  return foo.value;
}

TEST(HloPassContextTest, WorksWithAssignOrReturn) {
  Foo foo{42};
  HloPassContext ctx;
  EXPECT_THAT(ReadFooValue(ctx), StatusIs(absl::StatusCode::kNotFound));

  ASSERT_OK(ctx.Set(foo));
  EXPECT_THAT(ReadFooValue(ctx), IsOkAndHolds(42));
}

}  // namespace
}  // namespace xla
