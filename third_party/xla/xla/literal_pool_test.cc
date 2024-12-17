/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/literal_pool.h"

#include "xla/literal_util.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

TEST(LiteralPoolTest, GetCanonicalLiteral) {
  LiteralPool pool;

  auto l0 = LiteralUtil::CreateR2({{1., 2.}, {3., 4.}});
  auto l1 = LiteralUtil::CreateR2({{2., 1.}, {4., 3.}});

  {  // Use nested scope to allow garbage collection below.
    auto cl0_0 = pool.GetCanonicalLiteral(l0);
    auto cl0_1 = pool.GetCanonicalLiteral(l0);
    ASSERT_EQ(cl0_0, cl0_1);

    auto cl1_0 = pool.GetCanonicalLiteral(l1);
    auto cl1_1 = pool.GetCanonicalLiteral(l1);
    ASSERT_NE(cl0_0, cl1_0);
    ASSERT_EQ(cl1_0, cl1_1);
  }

  ASSERT_EQ(pool.GarbageCollect(), 2);
}

}  // namespace
}  // namespace xla
