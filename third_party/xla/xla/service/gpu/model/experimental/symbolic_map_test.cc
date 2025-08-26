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

#include "xla/service/gpu/model/experimental/symbolic_map.h"

#include <vector>

#include <gtest/gtest.h>
#include "xla/service/gpu/model/experimental/symbolic_expr.h"

namespace xla {
namespace gpu {
namespace {

TEST(SymbolicMapTest, IsEmpty) {
  SymbolicExprContext ctx;
  EXPECT_TRUE(SymbolicMap(0, 0, {}).IsEmpty());
  EXPECT_TRUE(SymbolicMap(2, 1, {}).IsEmpty());
  EXPECT_FALSE(SymbolicMap(1, 0, {ctx.CreateVariable(0)}).IsEmpty());
}

TEST(SymbolicMapTest, IsIdentity) {
  SymbolicExprContext ctx;

  // True identity
  EXPECT_TRUE(SymbolicMap(2, 0, {ctx.CreateVariable(0), ctx.CreateVariable(1)})
                  .IsIdentity());
  // True identity with symbols
  EXPECT_TRUE(SymbolicMap(2, 1, {ctx.CreateVariable(0), ctx.CreateVariable(1)})
                  .IsIdentity());

  // False: Wrong number of results
  EXPECT_FALSE(SymbolicMap(2, 0, {ctx.CreateVariable(0)}).IsIdentity());
  EXPECT_FALSE(SymbolicMap(1, 0, {ctx.CreateVariable(0), ctx.CreateVariable(1)})
                   .IsIdentity());

  // False: Wrong expression type
  EXPECT_FALSE(SymbolicMap(2, 0, {ctx.CreateVariable(0), ctx.CreateConstant(1)})
                   .IsIdentity());

  // False: Wrong variable ID
  EXPECT_FALSE(SymbolicMap(2, 0, {ctx.CreateVariable(1), ctx.CreateVariable(0)})
                   .IsIdentity());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
