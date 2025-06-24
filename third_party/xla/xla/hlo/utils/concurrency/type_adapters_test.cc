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

#include "xla/hlo/utils/concurrency/type_adapters.h"

#include <functional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"

namespace xla::concurrency {
namespace {
using ::testing::ElementsAreArray;

int call_fun(std::function<int()> f) { return f(); }

TEST(TurnMoveOnlyToCopyableWithCachingTest, CanCopyAssign) {
  const int kVal = -42;
  absl::AnyInvocable<int() &&> my_fun = []() { return kVal; };

  auto copyable_my_fun =
      TurnMoveOnlyToCopyableWithCaching<int>(std::move(my_fun));
  EXPECT_EQ(copyable_my_fun(), kVal);

  auto my_fun_copy = copyable_my_fun;
  EXPECT_EQ(copyable_my_fun(), kVal);
}

TEST(TurnMoveOnlyToCopyableWithCachingTest, CanCaptureCopyable) {
  const int kVal = -42;
  absl::AnyInvocable<int() &&> my_fun = []() { return kVal; };

  EXPECT_EQ(call_fun([f = TurnMoveOnlyToCopyableWithCaching<int>(
                          std::move(my_fun))]() mutable { return f(); }),
            kVal);
}

TEST(TurnMoveOnlyToCopyableWithCachingTest, VectorWrappingWrapsEachElement) {
  const int kVal0 = 42;
  const int kVal1 = 77;

  std::vector<absl::AnyInvocable<int() &&>> funs;
  funs.push_back([]() { return kVal0; });
  funs.push_back([]() { return kVal1; });

  std::vector<int> call0;
  std::vector<int> call1;
  for (auto& f :
       TurnMoveOnlyToCopyableWithCaching<int>::FromVector(std::move(funs))) {
    call0.push_back(f());
    call1.push_back(f());
  }

  EXPECT_THAT(call0, ElementsAreArray({kVal0, kVal1}));
  EXPECT_THAT(call1, ElementsAreArray({kVal0, kVal1}));
}
}  // namespace
}  // namespace xla::concurrency
